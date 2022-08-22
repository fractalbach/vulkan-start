// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
#pragma once

#include "vk_engine.h"
#include "vk_mesh.h"
#include <glm/glm.hpp>
#include <vk_types.h>

#include <unordered_map>
#include <vector>

const unsigned int FRAME_OVERLAP = 2; // number of frames to buffer

struct MeshPushConstants {
  glm::vec4 data;
  glm::mat4 render_matrix;
};

struct Material {
  VkPipeline pipeline;
  VkPipelineLayout pipelineLayout;
};

struct RenderObject {
  Mesh *mesh;
  Material *material;
  glm::mat4 transformMatrix;
};

struct GPUCameraData {
  glm::mat4 view;
  glm::mat4 proj;
  glm::mat4 viewproj;
};

struct FrameData {
  VkSemaphore _presentSemaphore;
  VkSemaphore _renderSemaphore;
  VkFence _renderFence;
  VkCommandPool _commandPool;
  VkCommandBuffer _mainCommandBuffer;
  AllocatedBuffer cameraBuffer;
  VkDescriptorSet globalDescriptor;
};

class VulkanEngine {
public:
  bool _isInitialized{false};
  int _frameNumber{0};
  int _selectedShader{0};
  int _pipelineCount{2};

  VkExtent2D _windowExtent{1600, 900};
  struct SDL_Window *_window{nullptr};

  void init();    // initializes everything in the engine
  void cleanup(); // shuts down the engine
  void draw();    // draw loop
  void run();     // run main loop

  VkInstance _instance;                      // Vulkan library handle
  VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
  VkPhysicalDevice _chosenGPU;               // GPU chosen as the default device
  VkDevice _device;                          // Vulkan device for commands
  VkSurfaceKHR _surface;                     // Vulkan window surface
  VmaAllocator _allocator;                   // Vulkan Memory Allocator

  // commands
  VkSwapchainKHR _swapchain;
  VkFormat _swapchainImageFormat; // format expected by the windowing system
  std::vector<VkImage> _swapchainImages;
  std::vector<VkImageView> _swapchainImageViews;

  VkQueue _graphicsQueue; // queue we will submit to
  uint32_t _graphicsQueueFamily;

  FrameData _frames[FRAME_OVERLAP];
  FrameData &get_current_frame();

  VkRenderPass _renderPass;
  std::vector<VkFramebuffer> _framebuffers;

  VkPipelineLayout _trianglePipelineLayout;

  VkPipeline _trianglePipeline;
  VkShaderModule _triangleVertexShader;
  VkShaderModule _triangleFragShader;

  VkPipeline _triangle2Pipeline;
  VkShaderModule _triangle2FragShader;
  VkShaderModule _triangle2VertexShader;

  VkPipeline _meshPipeline;
  VkPipelineLayout _meshPipelineLayout;
  VkShaderModule _triangle3VertexShader;
  Mesh _triangleMesh;

  Mesh _monkeyMesh;

  VkImageView _depthImageView;
  AllocatedImage _depthImage;
  VkFormat _depthFormat;

  VkDescriptorSetLayout _globalSetLayout;
  VkDescriptorPool _descriptorPool;

  std::vector<RenderObject> _renderables;
  std::unordered_map<std::string, Material> _materials;
  std::unordered_map<std::string, Mesh> _meshes;

  // Creates a new material and adds it to _materials map.
  Material *create_material(VkPipeline pipeline, VkPipelineLayout layout,
                            const std::string &name);

  // returns nullptr if not found
  Material *get_material(const std::string &name);

  // returns nullptr if not found
  Mesh *get_mesh(const std::string &name);

  void draw_objects(VkCommandBuffer cmd, RenderObject *first, int count);

  AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage,
                                VmaMemoryUsage memoryUsage);

private:
  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_default_renderpass();
  void init_frame_buffers();
  void init_sync_structures();
  void init_descriptors();
  void init_pipelines();
  void init_scene();

  bool load_shader_module(const char *filepath,
                          VkShaderModule *outShaderModule);

  void load_meshes();
  void upload_mesh(Mesh &mesh);

  std::vector<AllocatedBuffer *> _allocatedBuffers;
};

class PipelineBuilder {
public:
  std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
  VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
  VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
  VkViewport _viewport;
  VkRect2D _scissor;
  VkPipelineRasterizationStateCreateInfo _rasterizer;
  VkPipelineColorBlendAttachmentState _colorBlendAttachment;
  VkPipelineMultisampleStateCreateInfo _multisampling;
  VkPipelineLayout _pipelineLayout;
  VkPipelineDepthStencilStateCreateInfo _depthStencil;

  VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};
