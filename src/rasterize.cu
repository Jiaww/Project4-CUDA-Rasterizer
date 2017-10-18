/**
* @file      rasterize.cu
* @brief     CUDA-accelerated rasterization pipeline.
* @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
* @date      2012-2016
* @copyright University of Pennsylvania & STUDENT
*/

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

#define Perspective_Correct_Toggle 1
#define BackFaceCulling_Toggle 1
#define K_Buffer_Toggle 0
#define Bilinear_Color_Filter_Toggle 1
#define Naive_Sort_Toggle 0

#define Alpha_Intensity 0.6f

RenderMode curr_Mode = r_Triangle;

//int counter = 0;
//float time_ap = 0, time_r = 0, time_f = 0, time_s = 0;

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType {
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
							// glm::vec3 col;
		glm::vec2 texcoord0;
		TextureData* dev_diffuseTex = NULL;
		int texWidth, texHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
		// ...
		int TexWidth, TexHeight;

		glm::vec4 K_buffer[4];
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

	struct KBuffer4 {
		float depths[4];
	};
}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int *dev_mutex = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

static KBuffer4  *dev_k_buffer = NULL;
/**
* Kernel that writes the image to the OpenGL PBO directly.
*/
__global__
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		glm::vec3 color;
		color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
		color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
		color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0.9;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}


// From Wikipedia: https://en.wikipedia.org/wiki/Bilinear_filtering Part "Sample Code"
__device__ __host__ glm::vec3 getBilinearFilteredPixelColor(TextureData* tex, glm::vec2 uv, int texWidth, int texHeight) {
	float u = uv.s * texWidth - 0.5f;
	float v = uv.t * texHeight - 0.5f;
	int x = glm::floor(u);
	int y = glm::floor(v);
	float u_ratio = u - x;
	float v_ratio = v - y;
	float u_opposite = 1 - u_ratio;
	float v_opposite = 1 - v_ratio;
	int i0 = 3 * (x + y * texWidth);
	int i1 = 3 * ((x + 1) + y * texWidth);
	int i2 = 3 * (x + (y + 1) * texWidth);
	int i3 = 3 * ((x + 1) + (y + 1) * texWidth);

	float red = (tex[i0] * u_opposite + tex[i1] * u_ratio) * v_opposite + (tex[i2] * u_opposite + tex[i3] * u_ratio) * v_ratio;
	float green = (tex[i0 + 1] * u_opposite + tex[i1 + 1] * u_ratio) * v_opposite + (tex[i2 + 1] * u_opposite + tex[i3 + 1] * u_ratio) * v_ratio;
	float blue = (tex[i0 + 2] * u_opposite + tex[i1 + 2] * u_ratio) * v_opposite + (tex[i2 + 2] * u_opposite + tex[i3 + 2] * u_ratio) * v_ratio;

	return glm::vec3(red, green, blue) / 255.0f;
}


/**
* Writes fragment colors to the framebuffer
*/

__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, RenderMode renderMode) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		//framebuffer[index] = fragmentBuffer[index].color;

		Fragment fgbuffer = fragmentBuffer[index];
		// TODO: add your fragment shader code here
		glm::vec3 normal = fgbuffer.eyeNor;
		glm::vec3 lightDir = glm::normalize(glm::vec3(-3.0f, 5.0f, 5.0f) - fgbuffer.eyePos);

		float lambertian = glm::clamp(glm::dot(normal, lightDir), 0.0f, 1.0f);
		float specular = 0.0f;

		if (lambertian > 0.0f) {
			glm::vec3 viewDir = glm::normalize(-fgbuffer.eyePos);

			//this is blinn phong
			glm::vec3 halfDir = glm::normalize(lightDir + viewDir);
			float specAngle = glm::clamp(glm::dot(halfDir, normal), 0.0f, 1.0f);
			specular = glm::pow(specAngle, 16.0f);
		}
		glm::vec3 ambientColor = glm::vec3(0.0f, 0.0f, 0.0f);
		glm::vec3 diffuseColor;
		glm::vec3 specColor = glm::vec3(1.0, 1.0, 1.0);
		
		
#if K_Buffer_Toggle
		float a = Alpha_Intensity;
		for (int i = 3; i > 0; i--) {
			if (fgbuffer.K_buffer[i].w == 1.0f)
				fgbuffer.K_buffer[i] = glm::vec4(0.0f,0.0f,0.0f,1.0f);
			fgbuffer.K_buffer[i-1] = glm::vec4((a * glm::vec3(fgbuffer.K_buffer[i-1]) + (1 - a)*glm::vec3(fgbuffer.K_buffer[i])), fgbuffer.K_buffer[i-1].w);
		}
		diffuseColor = glm::vec3(fgbuffer.K_buffer[0]);
		specular = 0;
#else
		if (fgbuffer.dev_diffuseTex != NULL)
#if Bilinear_Color_Filter_Toggle
			diffuseColor = getBilinearFilteredPixelColor(fgbuffer.dev_diffuseTex, fgbuffer.texcoord0, fgbuffer.TexWidth, fgbuffer.TexHeight);
#else
			diffuseColor = fgbuffer.color;
#endif
		else
			diffuseColor = fgbuffer.color;
#endif
		glm::vec3 colorLinear = ambientColor + lambertian * diffuseColor + specular * specColor;
		framebuffer[index] = colorLinear;
		if(renderMode == r_Point || renderMode == r_Line)
			framebuffer[index] = diffuseColor;
	}
}

/**
* Called once at the beginning of the program to allocate memory.
*/
void rasterizeInit(int w, int h) {
	width = w;
	height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	cudaFree(dev_framebuffer);
	cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));
#if K_Buffer_Toggle
	cudaFree(dev_k_buffer);
	cudaMalloc((void**)&dev_k_buffer, width * height * sizeof(KBuffer4));
#endif
	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex, width * height * sizeof(int));

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}

__global__
void initKBuffer4(int w, int h, KBuffer4 * k_buffer)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		k_buffer[index].depths[0] = 1.0f;
		k_buffer[index].depths[1] = 1.0f;
		k_buffer[index].depths[2] = 1.0f;
		k_buffer[index].depths[3] = 1.0f;
	}
}

__global__
void initKBufferInFrag(int w, int h, Fragment * fragbuffer)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		fragbuffer[index].K_buffer[0].w = 1.0f;
		fragbuffer[index].K_buffer[1].w = 1.0f;
		fragbuffer[index].K_buffer[2].w = 1.0f;
		fragbuffer[index].K_buffer[3].w = 1.0f;
	}
}

/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {

	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {

			dev_dst[count * componentTypeByteSize * n
				+ offset * componentTypeByteSize
				+ j]

				=

				dev_src[byteOffset
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride)
				+ offset * componentTypeByteSize
				+ j];
		}
	}


}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {

	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	}
	else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode(
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
)
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);

									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();

					// Transform from local to camera
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}


	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}


	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());

		//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



__global__
void _vertexTransformAndAssembly(
	int numVertices,
	PrimitiveDevBufPointers primitive,
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal,
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		glm::vec4 position = glm::vec4(primitive.dev_position[vid], 1.0f);
		glm::vec4 clipVPosition = MVP * position;
		// Then divide the pos by its w element to transform into NDC space
		clipVPosition /= clipVPosition.w;
		// Finally transform x and y to viewport space
		clipVPosition.x = 0.5f * (float)width * (clipVPosition.x + 1.0f);  // Viewport(Screen / Window) Space
		clipVPosition.y = 0.5f * (float)height * (1.0f - clipVPosition.y); // Viewport(Screen / Window) Space 

		primitive.dev_verticesOut[vid].pos = clipVPosition;

		primitive.dev_verticesOut[vid].eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);
		glm::vec4 eyeSpacePos = (MV * glm::vec4(primitive.dev_position[vid], 1.0f));
		eyeSpacePos /= eyeSpacePos.w;
		primitive.dev_verticesOut[vid].eyePos = glm::vec3(eyeSpacePos);

		if (primitive.dev_diffuseTex != NULL) {
			primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
			primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
			primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
			primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
		}

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array

	}
}



static int curPrimitiveBeginId = 0;

__global__
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}

		// TODO: other primitive types (point, line)
	}

}

__device__ float cuda_clamp(float input, float min, float max) {
	float result = input;
	if (result < min)
		result = min;
	if (result > max)
		result = max;
	return result;
}
__device__ __host__ float cuda_getPerspectiveCorrectZ(glm::vec3 tri[3], glm::vec3 baryvalue) {
	float inverse_Z = baryvalue.x / (tri[0].z + FLT_EPSILON) + baryvalue.y / (tri[1].z + FLT_EPSILON) + baryvalue.z / (tri[2].z + FLT_EPSILON);
	return 1.0f / inverse_Z;
}
__device__ __host__ glm::vec2 cuda_getPerspectiveCorrectUV(glm::vec2 tri_uvs[3], glm::vec3 tri[3], glm::vec3 baryvalue, float Z) {
	glm::vec2 correct_texcoords = Z * glm::vec2(
		baryvalue.x * tri_uvs[0] / (tri[0].z + FLT_EPSILON) +
		baryvalue.y * tri_uvs[1] / (tri[1].z + FLT_EPSILON) +
		baryvalue.z * tri_uvs[2] / (tri[2].z + FLT_EPSILON));
	return correct_texcoords;
}
__device__ __host__ glm::vec3 cuda_getPerspectiveCorrectNormal(glm::vec3 tri_normals[3], glm::vec3 tri[3], glm::vec3 baryvalue, float Z) {
	glm::vec3 correct_normal = glm::normalize(Z * glm::vec3(
		baryvalue.x * tri_normals[0] / (tri[0].z + FLT_EPSILON) +
		baryvalue.y * tri_normals[1] / (tri[1].z + FLT_EPSILON) +
		baryvalue.z * tri_normals[2] / (tri[2].z + FLT_EPSILON)));
	return correct_normal;
}

__host__ __device__ void naive_sort(glm::vec4 *k_buffer4) {
	for (int i = 0; i < 3; i++) {
		float min = k_buffer4[i].w;
		int n = i;
		for (int j = i + 1; j < 4; j++) {
			if (k_buffer4[j].w < min) {
				n = j;
				min = k_buffer4[j].w;
			}
		}
		glm::vec4 temp = k_buffer4[i];
		k_buffer4[i] = k_buffer4[n];
		k_buffer4[n] = temp;
	}
}

__global__ void rasterizer(Fragment *fragmentBuffer, Primitive *primitives, int *depth, int num_primitives, int height, int width, int *mutex, KBuffer4* k_buffer) {
	// index of primitives
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pid < num_primitives) {
		Primitive this_primitives = primitives[pid];
#if BackFaceCulling_Toggle
		if (glm::dot(this_primitives.v[0].eyeNor, -this_primitives.v[0].eyePos) < 0.0f)
			return;
#endif
		glm::vec3 tri[3];
		//tri[0] = glm::vec3(0.5f - this_primitives.v[0].pos[0] * 3.0f, 0.8f - this_primitives.v[0].pos[1] * 3.0f, this_primitives.v[0].pos[2]);
		//tri[1] = glm::vec3(0.5f - this_primitives.v[1].pos[0] * 3.0f, 0.8f - this_primitives.v[1].pos[1] * 3.0f, this_primitives.v[1].pos[2]);
		//tri[2] = glm::vec3(0.5f - this_primitives.v[2].pos[0] * 3.0f, 0.8f - this_primitives.v[2].pos[1] * 3.0f, this_primitives.v[2].pos[2]);
		tri[0] = glm::vec3(this_primitives.v[0].pos);
		tri[1] = glm::vec3(this_primitives.v[1].pos);
		tri[2] = glm::vec3(this_primitives.v[2].pos);

		AABB bbox;
		bbox = getAABBForTriangle(tri);

		/*bbox.min.x *= width;
		bbox.max.x *= width;
		bbox.min.y *= height;
		bbox.max.y *= height;*/

		//clamp inside of the screen
		bbox.min.x = glm::clamp(bbox.min.x, 0.0f, float(width));
		bbox.max.x = glm::clamp(bbox.max.x, 0.0f, float(width));
		bbox.min.y = glm::clamp(bbox.min.y, 0.0f, float(height));
		bbox.max.y = glm::clamp(bbox.max.y, 0.0f, float(height));


		//scan the pixels inside of the bbox
		for (int i = bbox.min.x; i <= bbox.max.x; i++)
			for (int j = bbox.min.y; j <= bbox.max.y; j++) {
				glm::vec2 point(i, j);
				glm::vec3 baryvalue = calculateBarycentricCoordinate(tri, point);
				if (isBarycentricCoordInBounds(baryvalue)) {
					int pixel_index = i + j*width;
					float ffragDepth;
#if Perspective_Correct_Toggle
					ffragDepth = cuda_getPerspectiveCorrectZ(tri, baryvalue);
#else
					ffragDepth = baryvalue[0] * this_primitives.v[0].pos[2] + baryvalue[1] * this_primitives.v[1].pos[2] + baryvalue[2] * this_primitives.v[2].pos[2];
#endif
					int ifragDepth = INT_MAX * ffragDepth;
					
#if K_Buffer_Toggle
					if (ffragDepth < k_buffer[pixel_index].depths[3]) {

						atomicExch(&k_buffer[pixel_index].depths[3], ffragDepth);
#else
					if (ifragDepth < depth[pixel_index]) {
						atomicMin(&depth[pixel_index], ifragDepth);
#endif
						bool isSet;
						do {
							isSet = (atomicCAS(&mutex[pixel_index], 0, 1) == 0);
							if (isSet) {
								// Critical section goes here.
								// The critical section MUST be inside the wait loop;
								// if it is afterward, a deadlock will occur.
								//fragmentBuffer[pixel_index].color = glm::vec3(1, 1, 1);
#if Perspective_Correct_Toggle
								glm::vec2 tri_uvs[3];
								tri_uvs[0] = this_primitives.v[0].texcoord0;
								tri_uvs[1] = this_primitives.v[1].texcoord0;
								tri_uvs[2] = this_primitives.v[2].texcoord0;
								fragmentBuffer[pixel_index].texcoord0 = cuda_getPerspectiveCorrectUV(tri_uvs, tri, baryvalue, ffragDepth);
								//fragmentBuffer[pixel_index].texcoord0 = baryvalue[0] * this_primitives.v[0].texcoord0 + baryvalue[1] * this_primitives.v[1].texcoord0 + baryvalue[2] * this_primitives.v[2].texcoord0;
#else
								fragmentBuffer[pixel_index].texcoord0 = baryvalue[0] * this_primitives.v[0].texcoord0 + baryvalue[1] * this_primitives.v[1].texcoord0 + baryvalue[2] * this_primitives.v[2].texcoord0;
#endif
#if Perspective_Correct_Toggle
								glm::vec3 tri_normals[3];
								tri_normals[0] = this_primitives.v[0].eyeNor;
								tri_normals[1] = this_primitives.v[1].eyeNor;
								tri_normals[2] = this_primitives.v[2].eyeNor;
								fragmentBuffer[pixel_index].eyeNor = cuda_getPerspectiveCorrectNormal(tri_normals, tri, baryvalue, ffragDepth);
#else
								fragmentBuffer[pixel_index].eyeNor = baryvalue[0] * this_primitives.v[0].eyeNor + baryvalue[1] * this_primitives.v[1].eyeNor + baryvalue[2] * this_primitives.v[2].eyeNor;
#endif
								fragmentBuffer[pixel_index].eyePos = baryvalue[0] * this_primitives.v[0].eyePos + baryvalue[1] * this_primitives.v[1].eyePos + baryvalue[2] * this_primitives.v[2].eyePos;

								fragmentBuffer[pixel_index].TexWidth = this_primitives.v[0].texWidth;
								fragmentBuffer[pixel_index].TexHeight = this_primitives.v[0].texHeight;



#if K_Buffer_Toggle
								if (this_primitives.v[0].dev_diffuseTex != NULL) {
									fragmentBuffer[pixel_index].dev_diffuseTex = this_primitives.v[0].dev_diffuseTex;
#if Bilinear_Color_Filter_Toggle
									fragmentBuffer[pixel_index].color = getBilinearFilteredPixelColor(this_primitives.v[0].dev_diffuseTex, fragmentBuffer[pixel_index].texcoord0, this_primitives.v[0].texWidth, this_primitives.v[0].texHeight);
#else
									fragmentBuffer[pixel_index].color = glm::vec3(1.0f, 1.0f, 1.0f);
#endif
									//fragmentBuffer[pixel_index].K_buffer[3] = glm::vec4(fragmentBuffer[pixel_index].texcoord0, 0.0f, ffragDepth);
								}
								else
									fragmentBuffer[pixel_index].color = glm::vec3(1, 1, 1);
								//K_buffer RBGZ
								fragmentBuffer[pixel_index].K_buffer[3] = glm::vec4(fragmentBuffer[pixel_index].color, ffragDepth);
								//sort fragment k-buffer
#if Naive_Sort_Toggle
								naive_sort(fragmentBuffer[pixel_index].K_buffer);
#else
								float keys[4] = { fragmentBuffer[pixel_index].K_buffer[0].w, fragmentBuffer[pixel_index].K_buffer[1].w, fragmentBuffer[pixel_index].K_buffer[2].w , fragmentBuffer[pixel_index].K_buffer[3].w };
								thrust::sort_by_key(thrust::device, keys, keys + 4, fragmentBuffer[pixel_index].K_buffer);
#endif
								
#else
								if (this_primitives.v[0].dev_diffuseTex != NULL) {
									fragmentBuffer[pixel_index].dev_diffuseTex = this_primitives.v[0].dev_diffuseTex;
									fragmentBuffer[pixel_index].color = glm::vec3(1.0f,1.0f,1.0f);
									//fragmentBuffer[pixel_index].color = getBilinearFilteredPixelColor(this_primitives.v[0].dev_diffuseTex, fragmentBuffer[pixel_index].texcoord0, this_primitives.v[0].texWidth, this_primitives.v[0].texHeight);
								}
								else
									fragmentBuffer[pixel_index].color = glm::vec3(1, 1, 1);
#endif
								//k_max_idx[pixel_index] = 3;
							}
							if (isSet) {
								mutex[pixel_index] = 0;
							}
						} while (!isSet);
					}
				}
			}
	}

}

__global__ void rasterizer_Line(Fragment *fragmentBuffer, Primitive *primitives, int *depth, int num_primitives, int height, int width, int *mutex) {
	// index of primitives
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pid < num_primitives) {
		Primitive this_primitives = primitives[pid];
		//if (glm::dot(this_primitives.v[0].eyeNor, -this_primitives.v[0].eyePos) < 0.0f)
		//	return;

		//3 edges for each triangle
		for (int i = 0; i < 3; i++) {
			VertexOut v_outs[2];
			v_outs[0] = this_primitives.v[i % 3];
			v_outs[1] = this_primitives.v[(i + 1) % 3];
			glm::vec3 v_start, v_end;
			v_start = glm::vec3(v_outs[0].pos);
			v_end = glm::vec3(v_outs[1].pos);
			v_start = glm::clamp(v_start, glm::vec3(0, 0, 0), glm::vec3(width, height, 1.0f));
			v_end = glm::clamp(v_end, glm::vec3(0, 0, 0), glm::vec3(width, height, 1.0f));
			glm::vec3 v_dir = glm::normalize(v_end - v_start);
			int j = 0;
			while (true) {
				glm::vec3  v_curr = v_start + v_dir * float(j);
				j++;
				if (glm::dot(v_end - v_curr, v_dir) < 0.0f)
					break;
				int px, py;
				px = v_curr.x;
				py = v_curr.y;
				int pixel_index = px + py*width;
				glm::vec2 baryvalue;
				baryvalue[0] = glm::length(v_curr - v_start) / glm::length(v_end - v_start);
				baryvalue[1] = 1.0f - baryvalue[0];
				//Get perspective Correct Z
				float ffragDepth = baryvalue[0] / v_start.z + baryvalue[1] / v_end.z;
				ffragDepth = 1.0f / ffragDepth;
				int ifragDepth = INT_MAX * ffragDepth;
				if (ifragDepth < depth[pixel_index]) {
					atomicMin(&depth[pixel_index], ifragDepth);
					bool isSet;
					do {
						isSet = (atomicCAS(&mutex[pixel_index], 0, 1) == 0);
						if (isSet) {
							//TexCoords(Perspective Correct)
							fragmentBuffer[pixel_index].texcoord0 = (ffragDepth*(
								v_outs[0].texcoord0 * baryvalue[0] / v_outs[0].pos.z +
								v_outs[1].texcoord0 * baryvalue[1] / v_outs[1].pos.z
								)
								);
							//Normals(Per. Cor.)
							fragmentBuffer[pixel_index].eyeNor = glm::normalize(ffragDepth*(
								v_outs[0].eyeNor * baryvalue[0] / v_outs[0].pos.z +
								v_outs[1].eyeNor * baryvalue[1] / v_outs[1].pos.z
								)
							);
							//EyePos
							fragmentBuffer[pixel_index].eyePos = baryvalue[0] * v_outs[0].eyePos + baryvalue[1] * v_outs[1].eyePos;
							//
							fragmentBuffer[pixel_index].TexWidth = v_outs[0].texWidth;
							fragmentBuffer[pixel_index].TexHeight = v_outs[0].texHeight;
							//Tex
							if (v_outs[0].dev_diffuseTex != NULL) {
								fragmentBuffer[pixel_index].dev_diffuseTex = v_outs[0].dev_diffuseTex;
								//fragmentBuffer[pixel_index].color = getBilinearFilteredPixelColor(this_primitives.v[0].dev_diffuseTex, fragmentBuffer[pixel_index].texcoord0, this_primitives.v[0].texWidth, this_primitives.v[0].texHeight);
							}
							else
								fragmentBuffer[pixel_index].color = glm::vec3(1, 1, 1);

						}
						if (isSet) {
							mutex[pixel_index] = 0;
						}
					} while (!isSet);
				}

			}
		}
	}

}

__global__ void rasterizer_Point(Fragment *fragmentBuffer, Primitive *primitives, int *depth, int num_primitives, int height, int width, int *mutex) {
	// index of primitives
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pid < num_primitives) {
		Primitive this_primitives = primitives[pid];
		//if (glm::dot(this_primitives.v[0].eyeNor, -this_primitives.v[0].eyePos) < 0.0f)
		//	return;

		//3 points for each triangle
		for (int i = 0; i < 3; i++) {
			int pixel_index = int(this_primitives.v[i].pos.x) + int(this_primitives.v[i].pos.y) * width;
			float ffragDepth = this_primitives.v[i].pos.z;
			int ifragDepth = INT_MAX * ffragDepth;
			if (ifragDepth < depth[pixel_index]) {
				atomicMin(&depth[pixel_index], ifragDepth);
				bool isSet;
				do {
					isSet = (atomicCAS(&mutex[pixel_index], 0, 1) == 0);
					if (isSet) {
						//TexCoords
						fragmentBuffer[pixel_index].texcoord0 = this_primitives.v[i].texcoord0;
						//Normals(Per. Cor.)
						fragmentBuffer[pixel_index].eyeNor = this_primitives.v[i].eyeNor;
						//EyePos
						fragmentBuffer[pixel_index].eyePos = this_primitives.v[i].eyePos;
						//
						fragmentBuffer[pixel_index].TexWidth = this_primitives.v[0].texWidth;
						fragmentBuffer[pixel_index].TexHeight = this_primitives.v[0].texHeight;
						//Tex
						if (this_primitives.v[0].dev_diffuseTex != NULL) {
							fragmentBuffer[pixel_index].dev_diffuseTex = this_primitives.v[0].dev_diffuseTex;
							//fragmentBuffer[pixel_index].color = getBilinearFilteredPixelColor(this_primitives.v[0].dev_diffuseTex, fragmentBuffer[pixel_index].texcoord0, this_primitives.v[0].texWidth, this_primitives.v[0].texHeight);
						}
						else
							fragmentBuffer[pixel_index].color = glm::vec3(1, 1, 1);

					}
					if (isSet) {
						mutex[pixel_index] = 0;
					}
				} while (!isSet);
			}

		}
	}
}

struct BackFaceCulling_Cmp {
	__host__ __device__ bool operator()(const Primitive &p) {
		glm::vec3 face_normal = glm::cross(glm::vec3(p.v[1].pos - p.v[0].pos), glm::vec3(p.v[2].pos - p.v[0].pos));
		glm::vec3 inverse_eye_dir = -p.v[0].eyePos;
		// 'cause the NDC to pixel mirror the vertices, thus the front face turns to be clockwise 
		return glm::dot(face_normal, inverse_eye_dir) > 0.0f;
	}
};


/**
* Perform rasterization.
*/
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
	int sideLength2d = 8;
	dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	/*float time_elapsed=0;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start,0);
*/
	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices,
						curPrimitiveBeginId,
						dev_primitives,
						*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}

	/*cudaEventRecord( stop,0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_elapsed,start,stop);
	if (counter < 100) {
		time_ap += time_elapsed;
	}
	else if (counter == 100) {
		printf("Vertex Process & primitive Assembly: %f ms\n", time_ap);
	}*/


	int Culled_totalNumPrimitives = totalNumPrimitives;
#if BackFaceCulling_Toggle
	Primitive* dev_primitives_end = thrust::remove_if(thrust::device, dev_primitives, dev_primitives + totalNumPrimitives, BackFaceCulling_Cmp());
	Culled_totalNumPrimitives = dev_primitives_end - dev_primitives;
	if (Culled_totalNumPrimitives <= 0)
		Culled_totalNumPrimitives = 1;
#endif
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
#if K_Buffer_Toggle
	initKBuffer4 << <blockCount2d, blockSize2d >> > (width, height, dev_k_buffer);
	initKBufferInFrag << <blockCount2d, blockSize2d >> > (width, height, dev_fragmentBuffer);
#endif
	//cudaEventRecord(start, 0);

	// TODO: rasterize
	cudaMemset(dev_mutex, 0, width * height * sizeof(int));
	dim3 numThreadsPerBlock(128);
	dim3 numBlocksForPrimitives((Culled_totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
	if (curr_Mode == r_Point)
		rasterizer_Point << <numBlocksForPrimitives, numThreadsPerBlock >> >(dev_fragmentBuffer, dev_primitives, dev_depth, Culled_totalNumPrimitives, height, width, dev_mutex);
	else if (curr_Mode == r_Line)
		rasterizer_Line << <numBlocksForPrimitives, numThreadsPerBlock >> >(dev_fragmentBuffer, dev_primitives, dev_depth, Culled_totalNumPrimitives, height, width, dev_mutex);
	else if (curr_Mode == r_Triangle)
		rasterizer << <numBlocksForPrimitives, numThreadsPerBlock >> >(dev_fragmentBuffer, dev_primitives, dev_depth, Culled_totalNumPrimitives, height, width, dev_mutex, dev_k_buffer);
	
	/*cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_elapsed, start, stop);
	if (counter < 100) {
		time_r += time_elapsed;
	}
	else if (counter == 100) {
		printf("Rasterization: %f ms\n", time_r);
	}*/

	checkCUDAError("rasterization");
	
	//cudaEventRecord(start, 0);

	// Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer, curr_Mode);
	
	/*cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_elapsed, start, stop);*/
	if (counter < 100) {
		time_f += time_elapsed;
	}
	else if (counter == 100) {
		printf("Render(Fragment Shader): %f ms\n", time_f);
	}

	//cudaEventRecord(start, 0);

	checkCUDAError("fragment shader");
	// Copy framebuffer into OpenGL buffer for OpenGL previewing
	
	sendImageToPBO << <blockCount2d, blockSize2d >> >(pbo, width, height, dev_framebuffer);
	
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(start);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&time_elapsed, start, stop);
	//if (counter < 100) {
	//	time_s += time_elapsed;
	//}
	//else if (counter == 100) {
	//	printf("SendToPBO: %f ms\n", time_s);
	//}
	//counter++;

	checkCUDAError("copy render result to pbo");
}

/**
* Called once at the end of the program to free CUDA memory.
*/
void rasterizeFree() {

	// deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);


			//TODO: release other attributes and materials
		}
	}

	////////////

	cudaFree(dev_primitives);
	dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

	cudaFree(dev_framebuffer);
	dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;
#if K_Buffer_Toggle
	cudaFree(dev_k_buffer);
	dev_k_buffer = NULL;

#endif
	cudaFree(dev_mutex);
	dev_mutex = NULL;

	checkCUDAError("rasterize Free");
}
