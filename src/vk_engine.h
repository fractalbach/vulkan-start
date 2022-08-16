﻿// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
#pragma once

#include <vector>

#include "vk_engine.h"
#include "vk_mesh.h"
#include <glm/glm.hpp>
#include <vk_types.h>

struct MeshPushConstants {
  glm::vec4 data;
  glm::mat4 render_matrix;
  glm::int32 time;
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
  VkCommandPool _commandPool;
  VkCommandBuffer _mainCommandBuffer; // buffer we record into

  VkRenderPass _renderPass;
  std::vector<VkFramebuffer> _framebuffers;

  VkSemaphore _presentSemaphore, _renderSemaphore;
  VkFence _renderFence;

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

private:
  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_default_renderpass();
  void init_frame_buffers();
  void init_sync_structures();
  void init_pipelines();

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
  VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};