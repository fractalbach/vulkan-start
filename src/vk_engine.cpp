﻿#include "vk_engine.h"
#include "VkBootstrap.h" // bootstrap library
#include "vk_mesh.h"
#include <SDL.h>
#include <SDL_vulkan.h>
#include <glm/gtx/transform.hpp>
#include <vk_initializers.h>
#include <vk_types.h>

#include <chrono>
#include <fstream>
#include <iostream>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

// macro used to check for errors, crash immediately, and then print a message
// for the user.
#define VK_CHECK(x)                                                            \
  do {                                                                         \
    VkResult err = x;                                                          \
    if (err) {                                                                 \
      std::cerr << "Vulkan Error: " << err << std::endl;                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)

//==============================================================================
// Initialize the Engine
//______________________________________________________________________________

void VulkanEngine::init() {
  // We initialize SDL and create a window with it.
  SDL_Init(SDL_INIT_VIDEO);

  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

  _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED,
                             SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                             _windowExtent.height, window_flags);

  init_vulkan();    // load the core Vulkan structures
  init_swapchain(); // creates the swapchain
  init_commands();  // creates graphics queue and command buffer
  init_default_renderpass();
  init_frame_buffers();
  init_sync_structures(); // VkFence and VkSemaphore
  init_pipelines();       // loads shader programs
  load_meshes();
  init_scene();

  _isInitialized = true; // everything went fine
}

//==============================================================================
// Initialize Vulkan
//______________________________________________________________________________

void VulkanEngine::init_vulkan() {

  vkb::InstanceBuilder builder;

  // make the Vulkan instance, with basic debug features
  auto inst_ret = builder.set_app_name("Example Vulkan App")
                      .request_validation_layers(true)
                      .require_api_version(1, 1, 0)
                      .use_default_debug_messenger()
                      .build();

  vkb::Instance vkb_inst = inst_ret.value();

  _instance = vkb_inst.instance;               // store the instance
  _debug_messenger = vkb_inst.debug_messenger; // store debug messsenger

  // get surface of the window we opened with SDL
  SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

  // use vkbootstrap to select a GPU. We want a GPU that can write to SDL
  // surface and which supports Vulkan API 1.1
  vkb::PhysicalDeviceSelector selector{vkb_inst};
  vkb::PhysicalDevice physicalDevice =
      selector.set_minimum_version(1, 1).set_surface(_surface).select().value();

  // create the final Vulkan device
  vkb::DeviceBuilder deviceBuilder{physicalDevice};
  vkb::Device vkbDevice = deviceBuilder.build().value();

  // Get the VkDevice handle used in the rest of a Vulkan app
  _device = vkbDevice.device;
  _chosenGPU = physicalDevice.physical_device;

  // use vkbootstrap to get a Graphics queue
  _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
  _graphicsQueueFamily =
      vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

  // Initialize the memory allocator
  VmaAllocatorCreateInfo allocatorInfo = {
      .physicalDevice = _chosenGPU,
      .device = _device,
      .instance = _instance,
  };
  vmaCreateAllocator(&allocatorInfo, &_allocator);
}

//==============================================================================
// Initialize Swapchain
//______________________________________________________________________________

void VulkanEngine::init_swapchain() {
  vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};

  vkb::Swapchain vkbSwapchain =
      swapchainBuilder.use_default_format_selection()
          .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR) // vsync
          .set_desired_extent(_windowExtent.width, _windowExtent.height)
          .build()
          .value();

  // store swapchain and its related images
  _swapchain = vkbSwapchain.swapchain;
  _swapchainImages = vkbSwapchain.get_images().value();
  _swapchainImageViews = vkbSwapchain.get_image_views().value();
  _swapchainImageFormat = vkbSwapchain.image_format;

  // depth image size will match the window
  VkExtent3D depthImageExtent = {_windowExtent.width, _windowExtent.height, 1};

  _depthFormat = VK_FORMAT_D32_SFLOAT;

  VkImageCreateInfo dimg_info = vkinit::image_create_info(
      _depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      depthImageExtent);

  VmaAllocationCreateInfo dimg_allocinfo = {
      .usage = VMA_MEMORY_USAGE_GPU_ONLY,
      .requiredFlags =
          VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
  };

  vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_depthImage._image,
                 &_depthImage._allocation, nullptr);

  VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(
      _depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

  VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));
}

//==============================================================================
// Initialize Command Pool containing individual Command Buffers
//______________________________________________________________________________

void VulkanEngine::init_commands() {
  // create a command pool for commands submitted to the graphics queue.
  // we also want the pool to allow for resetting of individual command buffers
  VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(
      _graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  VK_CHECK(
      vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

  // allocate the default command buffer that we will use for rendering
  VkCommandBufferAllocateInfo cmdAllocInfo =
      vkinit::command_buffer_allocate_info(_commandPool, 1);

  VK_CHECK(
      vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_mainCommandBuffer));
}

//==============================================================================
// Initialize Renderpass
//______________________________________________________________________________

void VulkanEngine::init_default_renderpass() {

  VkAttachmentDescription color_attachment = {
      .format = _swapchainImageFormat,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
  };

  // ------------------------------------------------------------------------

  VkAttachmentDescription depth_attachment = {
      .flags = 0,
      .format = _depthFormat,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
  };

  VkAttachmentReference depth_attachment_ref = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
  };

  VkAttachmentDescription attachments[2] = {color_attachment, depth_attachment};

  // ------------------------------------------------------------------------

  VkSubpassDependency dependency = {
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
  };

  VkSubpassDependency depth_dependency = {
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
  };

  VkSubpassDependency dependencies[2] = {dependency, depth_dependency};

  // ------------------------------------------------------------------------

  // attachment number will index into the pAttachments array in the parent
  // renderpass itself
  VkAttachmentReference color_attachment_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
  };

  // we are going to create 1 subpass, which is the minimum you can do
  VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_attachment_ref,
      .pDepthStencilAttachment = &depth_attachment_ref,
  };

  // ------------------------------------------------------------------------

  // Now that the subpass is done, we can actually create the Renderpass
  VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2,
      .pAttachments = &attachments[0],
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 2,
      .pDependencies = &dependencies[0],
  };

  VK_CHECK(
      vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));
}

//==============================================================================
// Initialize Frame Buffers
//______________________________________________________________________________

void VulkanEngine::init_frame_buffers() {
  // create the framebuffers for the swapchain images. This will connect the
  // render-pass to the images for rendering
  VkFramebufferCreateInfo fb_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .pNext = nullptr,
      .renderPass = _renderPass,
      .attachmentCount = 1,
      .width = _windowExtent.width,
      .height = _windowExtent.height,
      .layers = 1,
  };

  // grab how many images we have in the swapchain
  const uint32_t swapchain_imagecount = _swapchainImages.size();
  _framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

  // create framebuffers for each of the swapchain image views
  for (int i = 0; i < swapchain_imagecount; i++) {
    VkImageView attachments[2] = {_swapchainImageViews[i], _depthImageView};
    fb_info.pAttachments = attachments;
    fb_info.attachmentCount = 2;
    VK_CHECK(
        vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));
  }
}

//==============================================================================
// Initialize Fences and Semaphores
//______________________________________________________________________________

void VulkanEngine::init_sync_structures() {

  VkFenceCreateInfo fenceCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .pNext = nullptr,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
  };

  VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));

  VkSemaphoreCreateInfo semaphoreCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
  };

  VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
                             &_presentSemaphore));

  VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
                             &_renderSemaphore));
}

//==============================================================================
// Initialize Pipelines
//______________________________________________________________________________

void VulkanEngine::init_pipelines() {

  // _________________________________________________________
  // Loading Shader Modules
  // ---------------------------------------------------------

  std::vector<std::pair<std::string, VkShaderModule *>> shaders = {
      {"shaders/triangle.vert.spv", &_triangleVertexShader},
      {"shaders/triangle2.vert.spv", &_triangle2VertexShader},
      {"shaders/triangle3.vert.spv", &_triangle3VertexShader},
      {"shaders/triangle.frag.spv", &_triangleFragShader},
      {"shaders/triangle2.frag.spv", &_triangle2FragShader},
  };

  for (auto x : shaders) {
    auto filename{x.first.c_str()};
    auto shader{x.second};
    if (!load_shader_module(filename, shader)) {
      std::cerr << "[ERROR]: Unable to load: " << filename << std::endl;
    } else {
      std::cout << "Loaded: " << filename << std::endl;
    }
  }

  // _________________________________________________________
  //  Layout: Original
  // ---------------------------------------------------------

  VkPipelineLayoutCreateInfo pipeline_layout_info =
      vkinit::pipeline_layout_create_info();

  VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr,
                                  &_trianglePipelineLayout));

  // _________________________________________________________
  // Layout: Mesh
  // ---------------------------------------------------------

  VkPipelineLayoutCreateInfo mesh_pipeline_layout_info =
      vkinit::pipeline_layout_create_info();

  VkPushConstantRange push_constant = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      .offset = 0,
      .size = sizeof(MeshPushConstants),
  };

  mesh_pipeline_layout_info.pPushConstantRanges = &push_constant;
  mesh_pipeline_layout_info.pushConstantRangeCount = 1;

  VK_CHECK(vkCreatePipelineLayout(_device, &mesh_pipeline_layout_info, nullptr,
                                  &_meshPipelineLayout));

  // _________________________________________________________
  // Pipeline Builder
  // ---------------------------------------------------------

  // build the stage-create-info for both vertex and fragment stages. This
  // lets the pipeline know the shader modules per stage
  PipelineBuilder pipelineBuilder;

  pipelineBuilder._shaderStages.push_back(
      vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT,
                                                _triangleVertexShader));

  pipelineBuilder._shaderStages.push_back(
      vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT,
                                                _triangleFragShader));

  // vertex input controls how to read vertices from vertex buffers. We aren't
  // using it yet
  pipelineBuilder._vertexInputInfo = vkinit::vertex_input_create_info();

  // input assembly is the configuration for drawing triangle lists, strips,
  // or individual points. we are just going to draw triangle list
  pipelineBuilder._inputAssembly =
      vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

  // build viewport and scissor from the swapchain extents
  pipelineBuilder._viewport.x = 0.0f;
  pipelineBuilder._viewport.y = 0.0f;
  pipelineBuilder._viewport.width = (float)_windowExtent.width;
  pipelineBuilder._viewport.height = (float)_windowExtent.height;
  pipelineBuilder._viewport.minDepth = 0.0f;
  pipelineBuilder._viewport.maxDepth = 1.0f;

  pipelineBuilder._scissor.offset = {0, 0};
  pipelineBuilder._scissor.extent = _windowExtent;

  // configure the rasterizer to draw filled triangles
  pipelineBuilder._rasterizer =
      vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);

  // we don't use multisampling, so just run the default one
  pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

  // a single blend attachment with no blending and writing to RGBA
  pipelineBuilder._colorBlendAttachment =
      vkinit::color_blend_attachment_state();

  // use the triangle layout we created
  pipelineBuilder._pipelineLayout = _trianglePipelineLayout;

  // default depthtesting
  pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(
      true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

  // finally build the pipeline
  _trianglePipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

  // _________________________________________________________
  //  Triangle 2 Pipeline!
  // ---------------------------------------------------------

  // clear the shader stages for the builder
  pipelineBuilder._shaderStages.clear();

  // add the other shaders
  pipelineBuilder._shaderStages.push_back(
      vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT,
                                                _triangle2VertexShader));
  pipelineBuilder._shaderStages.push_back(
      vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT,
                                                _triangle2FragShader));

  // build the triangle 2 pipeline
  _triangle2Pipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

  // default depthtesting
  pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(
      true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

  // _________________________________________________________
  //  Mesh PIpeline!
  // ---------------------------------------------------------
  VertexInputDescription vertDescription = Vertex::get_vertex_description();

  pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions =
      vertDescription.attributes.data();

  pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount =
      vertDescription.attributes.size();

  pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions =
      vertDescription.bindings.data();

  pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount =
      vertDescription.bindings.size();

  pipelineBuilder._shaderStages.clear();

  pipelineBuilder._shaderStages.push_back(
      vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT,
                                                _triangle3VertexShader));

  pipelineBuilder._shaderStages.push_back(
      vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT,
                                                _triangle2FragShader));

  pipelineBuilder._pipelineLayout = _meshPipelineLayout;

  _meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

  create_material(_meshPipeline, _meshPipelineLayout, "defaultmesh");

  // _________________________________________________________
  //  Free The Shader Modules
  // ---------------------------------------------------------
  vkDestroyShaderModule(_device, _triangleVertexShader, nullptr);
  vkDestroyShaderModule(_device, _triangleFragShader, nullptr);
  vkDestroyShaderModule(_device, _triangle2VertexShader, nullptr);
  vkDestroyShaderModule(_device, _triangle2FragShader, nullptr);
  vkDestroyShaderModule(_device, _triangle3VertexShader, nullptr);
}

//==============================================================================
// Initialize Scene
//
// Loads up render objects into an array to enable the engine to draw them
//______________________________________________________________________________

void VulkanEngine::init_scene() {

  RenderObject monkey = {
      .mesh = get_mesh("monkey"),
      .material = get_material("defaultmesh"),
      .transformMatrix = glm::mat4(1.0f),
  };

  _renderables.push_back(monkey);

  // creates 20 triangles that surround the monkey!
  for (int x = -20; x <= 20; x++) {
    for (int y = -20; y <= 20; y++) {

      glm::mat4 translation =
          glm::translate(glm::mat4{1.0}, glm::vec3(x, 0, y));

      glm::mat4 scale = glm::scale(glm::mat4(1.0), glm::vec3(0.2, 0.2, 0.2));

      RenderObject tri = {
          .mesh = get_mesh("triangle"),
          .material = get_material("defaultmesh"),
          .transformMatrix = translation * scale,
      };

      _renderables.push_back(tri);
    }
  }
}

//==============================================================================

void VulkanEngine::cleanup() {
  if (_isInitialized) {

    vkQueueWaitIdle(_graphicsQueue);

    vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
    for (auto x : _allocatedBuffers) {
      vmaDestroyBuffer(_allocator, x->_buffer, x->_allocation);
    }

    vmaDestroyAllocator(_allocator);

    vkDestroyPipeline(_device, _trianglePipeline, nullptr);
    vkDestroyPipeline(_device, _triangle2Pipeline, nullptr);
    vkDestroyPipeline(_device, _meshPipeline, nullptr);
    vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
    vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);

    vkDestroySemaphore(_device, _renderSemaphore, nullptr);
    vkDestroySemaphore(_device, _presentSemaphore, nullptr);
    vkDestroyFence(_device, _renderFence, nullptr);

    vkDestroyCommandPool(_device, _commandPool, nullptr);
    vkDestroyImageView(_device, _depthImageView, nullptr);
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);
    vkDestroyRenderPass(_device, _renderPass, nullptr);

    for (int i = 0; i < _framebuffers.size(); i++) {
      vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
      vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
    }

    vkDestroyDevice(_device, nullptr);
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
    vkDestroyInstance(_instance, nullptr);
    SDL_DestroyWindow(_window);

    std::cout << "Cleanup Completed." << std::endl;
  }
}

//==============================================================================

void VulkanEngine::draw() {
  // wait until the GPU has finished rendering the last frame. Timeout of 1
  // second
  VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
  VK_CHECK(vkResetFences(_device, 1, &_renderFence));

  // request image from the swapchain, one second timeout
  uint32_t swapchainImageIndex;
  VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000,
                                 _presentSemaphore, nullptr,
                                 &swapchainImageIndex));

  // now that we are sure that the commands finished executing, we can safely
  // reset the command buffer to begin recording again.
  VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));

  // naming it cmd for shorter writing
  VkCommandBuffer cmd = _mainCommandBuffer;

  // begin the command buffer recording. We will use this command buffer
  // exactly once, so we want to let Vulkan know that
  VkCommandBufferBeginInfo cmdBeginInfo = {};
  cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cmdBeginInfo.pNext = nullptr;

  cmdBeginInfo.pInheritanceInfo = nullptr;
  cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

  // make a clear-color from frame number. This will flash with a 120*pi frame
  // period.
  VkClearValue clearValue;
  // float flash = abs(sin(_frameNumber / (5 * 60.f)));
  float flash = 0.5;
  clearValue.color = {{0.0f, 0.0f, flash, 1.0f}};

  // clear depth at 1
  VkClearValue depthClear;
  depthClear.depthStencil.depth = 1.0f;

  VkClearValue clearValues[] = {clearValue, depthClear};

  // start the main renderpass. We will use the clear color from above, and
  // the framebuffer of the index the swapchain gave us
  VkRenderPassBeginInfo rpInfo = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .pNext = nullptr,
      .renderPass = _renderPass,
      .framebuffer = _framebuffers[swapchainImageIndex],
      .clearValueCount = 2,
      .pClearValues = &clearValues[0],
  };
  rpInfo.renderArea.offset = {.x = 0, .y = 0};
  rpInfo.renderArea.extent = _windowExtent;

  // finishes the rendering, and transitions to we image we specified, which
  // is "ready to be dislayed"
  VkDeviceSize offset = 0;

  // glm::vec3 camPos = {0.f, 0.f, -2.f};

  // glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);

  // glm::mat4 projection =
  //     glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.0f);

  // projection[1][1] *= -1;

  // glm::mat4 model = glm::rotate(
  //     glm::mat4{1.0f}, glm::radians(_frameNumber * 0.4f), glm::vec3(0, 1,
  //     0));

  // glm::mat4 mesh_matrix = projection * view * model;

  // auto current_time = std::chrono::system_clock::now().time_since_epoch();
  // int time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(current_time)
  //                .count();

  // MeshPushConstants constants{
  //     .data = glm::vec4{glm::float32((-time % 10000)), 0.0, 0.0, 0.0},
  //     .render_matrix = mesh_matrix,
  //     .time = time,
  // };

  vkCmdBindVertexBuffers(cmd, 0, 1, &_monkeyMesh._vertexBuffer._buffer,
                         &offset);

  vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

  draw_objects(cmd, _renderables.data(), _renderables.size());

  // switch (_selectedShader) {
  // case 1:
  //   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
  //   _triangle2Pipeline); vkCmdDraw(cmd, 3, 1, 0, 0); break;
  // case 2:
  //   vkCmdPushConstants(cmd, _meshPipelineLayout,
  //                      VK_SHADER_STAGE_FRAGMENT_BIT |
  //                          VK_SHADER_STAGE_VERTEX_BIT,
  //                      0, sizeof(MeshPushConstants), &constants);

  //   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline);

  //   vkCmdBindVertexBuffers(cmd, 0, 1, &_monkeyMesh._vertexBuffer._buffer,
  //                          &offset);

  //   vkCmdDraw(cmd, _monkeyMesh._vertices.size(), 1, 0, 0);
  //   break;
  // default:
  //   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
  //   _trianglePipeline); vkCmdDraw(cmd, 3, 1, 0, 0);
  // }

  // finalize the render pass
  vkCmdEndRenderPass(cmd);

  // finalize the command buffer (we can no longer add commands, but it can
  // now be executed)
  VK_CHECK(vkEndCommandBuffer(cmd));

  // -------------------------------------------------------------------------

  // prepare the submission to the queue. we want to wait on the
  // _presentSemaphore, as that semaphore is signaled when the swapchain is
  // ready we will signal the _renderSemaphore, to signal that rendering has
  // finished

  VkPipelineStageFlags waitStage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

  VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &_presentSemaphore,
      .pWaitDstStageMask = &waitStage,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &_renderSemaphore,
  };

  // submit command buffer to the queue and execute it. _renderFence will now
  // block until the graphic commands finish execution
  VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

  // -------------------------------------------------------------------------

  // this will put the image we just rendered into the visible window. we want
  // to wait on the _renderSemaphore for that, as it's necessary that drawing
  // commands have finished before the image is displayed to the user
  VkPresentInfoKHR presentInfo = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .pNext = nullptr,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &_renderSemaphore,
      .swapchainCount = 1,
      .pSwapchains = &_swapchain,
      .pImageIndices = &swapchainImageIndex,
  };

  // DISPLAYS AN IMAGE ON THE SCREEN!!!
  VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

  // increase the number of frames drawn
  _frameNumber++;
}

//==============================================================================

void VulkanEngine::run() {
  SDL_Event e;
  bool bQuit = false;

  // main loop
  while (!bQuit) {
    // Handle events on queue
    while (SDL_PollEvent(&e) != 0) {
      // close the window when user alt-f4s or clicks the X button
      if (e.type == SDL_QUIT) {
        bQuit = true;
      } else if (e.type == SDL_KEYDOWN) {
        if (e.key.keysym.sym == SDLK_SPACE) {
          _selectedShader = (_selectedShader + 1) % (_pipelineCount + 1);
        }
      }
    }

    draw();
  }
}

//==============================================================================

bool VulkanEngine::load_shader_module(const char *filepath,
                                      VkShaderModule *outShaderModule) {

  std::ifstream file(filepath, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "cannot open file: " << filepath << std::endl;
    return false;
  }
  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
  file.seekg(0);
  file.read((char *)buffer.data(), fileSize);
  file.close();

  // create a new shader module
  VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .pNext = nullptr,
      .codeSize = buffer.size() * sizeof(uint32_t),
      .pCode = buffer.data(),
  };

  VkShaderModule shaderModule;
  bool outcome = vkCreateShaderModule(_device, &info, nullptr, &shaderModule);
  if (outcome != VK_SUCCESS) {
    std::cerr << "error: could not create shader module: " << filepath
              << std::endl;
    return false;
  }
  *outShaderModule = shaderModule;
  return true;
}

//==============================================================================

void VulkanEngine::load_meshes() {
  _triangleMesh._vertices.resize(3);

  // vertex positions
  _triangleMesh._vertices[0].position = {1.f, 1.f, 0.5f};
  _triangleMesh._vertices[1].position = {-1.f, 1.f, 0.5f};
  _triangleMesh._vertices[2].position = {0.f, -1.f, 0.5f};

  // vertex colors, all green
  _triangleMesh._vertices[0].color = {0.f, 1.f, 0.0f}; // pure green
  _triangleMesh._vertices[1].color = {0.f, 1.f, 0.0f}; // pure green
  _triangleMesh._vertices[2].color = {0.f, 1.f, 0.0f}; // pure green

  // monkey mesh!
  _monkeyMesh.load_from_obj("assets/monkey_smooth.obj");

  // sends the mesh data to the GPU
  upload_mesh(_triangleMesh);
  upload_mesh(_monkeyMesh);

  _meshes["monkey"] = _monkeyMesh;
  _meshes["triangle"] = _triangleMesh;
}

//==============================================================================

void VulkanEngine::upload_mesh(Mesh &mesh) {

  // allocate vertex buffer
  VkBufferCreateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = mesh._vertices.size() * sizeof(Vertex),
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
  };

  // allow CPU to write and allow GPU to read
  VmaAllocationCreateInfo vmaallocInfo = {
      .usage = VMA_MEMORY_USAGE_CPU_TO_GPU,
  };

  // keep track of the allocated buffers so you can free them later
  _allocatedBuffers.push_back(&mesh._vertexBuffer);

  VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo,
                           &mesh._vertexBuffer._buffer,
                           &mesh._vertexBuffer._allocation, nullptr));

  // copy Vertex data
  void *data;
  vmaMapMemory(_allocator, mesh._vertexBuffer._allocation, &data);
  memcpy(data, mesh._vertices.data(), mesh._vertices.size() * sizeof(Vertex));
  vmaUnmapMemory(_allocator, mesh._vertexBuffer._allocation);
}

//==============================================================================

VkPipeline PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass) {
  VkPipelineViewportStateCreateInfo state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .viewportCount = 1,
      .pViewports = &_viewport,
      .scissorCount = 1,
      .pScissors = &_scissor,
  };

  VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .pNext = nullptr,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_COPY,
      .attachmentCount = 1,
      .pAttachments = &_colorBlendAttachment,
  };

  VkGraphicsPipelineCreateInfo pipe = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .stageCount = static_cast<uint32_t>(_shaderStages.size()),
      .pStages = _shaderStages.data(),
      .pVertexInputState = &_vertexInputInfo,
      .pInputAssemblyState = &_inputAssembly,
      .pViewportState = &state,
      .pRasterizationState = &_rasterizer,
      .pMultisampleState = &_multisampling,
      .pDepthStencilState = &_depthStencil,
      .pColorBlendState = &blend,
      .layout = _pipelineLayout,
      .renderPass = pass,
      .subpass = 0,
      .basePipelineHandle = VK_NULL_HANDLE,
  };

  VkPipeline newPipeline;
  VkResult res = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipe,
                                           nullptr, &newPipeline);
  if (res != VK_SUCCESS) {
    std::cerr << "error: failed to create graphics pipeline" << std::endl;
    return VK_NULL_HANDLE;
  } else {
    return newPipeline;
  }
}

//==============================================================================
// Materials and Mesh Storage Management
//______________________________________________________________________________

Material *VulkanEngine::create_material(VkPipeline pipeline,
                                        VkPipelineLayout layout,
                                        const std::string &name) {
  Material mat = {
      .pipeline = pipeline,
      .pipelineLayout = layout,
  };
  _materials[name] = mat;
  return &_materials[name];
}

// returns nullptr if not found
Material *VulkanEngine::get_material(const std::string &name) {
  auto it = _materials.find(name);
  if (it == _materials.end()) {
    return nullptr;
  } else {
    return &(*it).second;
  }
}

// returns nullptr if not found
Mesh *VulkanEngine::get_mesh(const std::string &name) {
  auto it = _meshes.find(name);
  if (it == _meshes.end()) {
    return nullptr;
  } else {
    return &(*it).second;
  }
}

//==============================================================================
// Drawing each Object
//______________________________________________________________________________

void VulkanEngine::draw_objects(VkCommandBuffer cmd, RenderObject *first,
                                int count) {


  // Bind each material and mesh to the pipeline, then tell gpu to draw.

  Mesh *lastMesh = nullptr;
  Material *lastMaterial = nullptr;

  for (int i = 0; i < count; i++) {

    // calculate the current time. This should happen before looping, to ensure
    // that each object has the same time value.
    auto t = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t);
    int time = -ms.count();



    // make model view matrix for rendering the object (camera view/projection)
    glm::vec3 camPos = {0.f, -6.f, (-40.f + 100 * ((time%10000)/10000.f))};
    glm::mat4 view = glm::translate(glm::mat4(1.0f), camPos);
    glm::mat4 projection =
        glm::perspective(glm::radians(70.f), (1700.f / 900.f), 0.1f, 200.f);
    projection[1][1] *= -1;

    glm::vec4 data = {glm::float32((time % 10000)), 0.0, 0.0, 0.0};

    RenderObject &object = first[i];

    // The pipeline might already have our material bound.
    if (object.material != lastMaterial) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        object.material->pipeline);
      lastMaterial = object.material;
    }

    glm::mat4 model = object.transformMatrix;
    glm::mat4 mesh_matrix = projection * view * model;

    MeshPushConstants constants{
        .data = data,
        .render_matrix = mesh_matrix,
        .time = time,
    };

    vkCmdPushConstants(cmd, object.material->pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants),
                       &constants);

    if (object.mesh != lastMesh) {
      VkDeviceSize offset = 0;
      auto buf = &object.mesh->_vertexBuffer._buffer;
      vkCmdBindVertexBuffers(cmd, 0, 1, buf, &offset);
      lastMesh = object.mesh;
    }

    vkCmdDraw(cmd, object.mesh->_vertices.size(), 1, 0, 0);
  }
}
