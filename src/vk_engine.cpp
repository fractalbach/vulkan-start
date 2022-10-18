#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <vk_initializers.h>
#include <vk_types.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <glm/gtx/transform.hpp>
#include <iostream>

#include "VkBootstrap.h" // bootstrap library
#include "vk_mesh.h"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

//==============================================================================
// VK_CHECK - macro used to loudly crash when errors happen
//______________________________________________________________________________

// check for errors, crash immediately, and print a message for the user
#define VK_CHECK(x)                                                            \
  do {                                                                         \
    VkResult err = x;                                                          \
    if (err) {                                                                 \
      std::cerr << "Vulkan Error: " << err << std::endl;                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)

//==============================================================================
// Initialize the Engine - the first function that gets called
//______________________________________________________________________________

void
VulkanEngine::init()
{
  // We initialize SDL and create a window with it.
  SDL_Init(SDL_INIT_VIDEO);

  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

  _window = SDL_CreateWindow("Vulkan Engine",
                             SDL_WINDOWPOS_UNDEFINED,
                             SDL_WINDOWPOS_UNDEFINED,
                             _windowExtent.width,
                             _windowExtent.height,
                             window_flags);

  init_vulkan();    // load the core Vulkan structures
  init_swapchain(); // creates the swapchain
  init_commands();  // creates graphics queue and command buffer
  init_default_renderpass();
  init_frame_buffers();
  init_sync_structures(); // VkFence and VkSemaphore
  init_descriptors();     // VkDescriptorSet objects
  init_pipelines();       // loads shader programs
  load_meshes();
  init_scene();

  _isInitialized = true; // everything went fine
}

//==============================================================================
// Initialize Vulkan
//______________________________________________________________________________

void
VulkanEngine::init_vulkan()
{
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
  vkb::PhysicalDeviceSelector selector{ vkb_inst };
  vkb::PhysicalDevice physicalDevice =
    selector.set_minimum_version(1, 1).set_surface(_surface).select().value();

  // create the final Vulkan device
  vkb::DeviceBuilder deviceBuilder{ physicalDevice };
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

  // Display information about the current device
  // This will include name and information we currently care about, such as
  // the alignment for the uniform buffer
  _gpuProperties = vkbDevice.physical_device.properties;
  const auto& g = _gpuProperties;
  std::cout << "[Device Information]\n"
            << "  Name: " << g.deviceName << "\n"
            << "  Minimum Buffer Alignment: "
            << g.limits.minUniformBufferOffsetAlignment << "\n";
}

//==============================================================================
// Initialize Swapchain
//______________________________________________________________________________

void
VulkanEngine::init_swapchain()
{
  vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU, _device, _surface };

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
  VkExtent3D depthImageExtent = { _windowExtent.width,
                                  _windowExtent.height,
                                  1 };

  _depthFormat = VK_FORMAT_D32_SFLOAT;

  VkImageCreateInfo dimg_info =
    vkinit::image_create_info(_depthFormat,
                              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                              depthImageExtent);

  VmaAllocationCreateInfo dimg_allocinfo = {
    .usage = VMA_MEMORY_USAGE_GPU_ONLY,
    .requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
  };

  vmaCreateImage(_allocator,
                 &dimg_info,
                 &dimg_allocinfo,
                 &_depthImage._image,
                 &_depthImage._allocation,
                 nullptr);

  VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(
    _depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

  VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));
}

//==============================================================================
// Initialize Command Pool containing individual Command Buffers
//______________________________________________________________________________

void
VulkanEngine::init_commands()
{
  // create a command pool for commands submitted to the graphics queue.
  // we also want the pool to allow for resetting of individual command buffers
  VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(
    _graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  for (int i = 0; i < FRAME_OVERLAP; i++) {
    VK_CHECK(vkCreateCommandPool(
      _device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

    // allocate the default command buffer that we will use for rendering
    VkCommandBufferAllocateInfo cmdAllocInfo =
      vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(
      _device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));
  }
}

//==============================================================================
// Initialize Renderpass
//______________________________________________________________________________

void
VulkanEngine::init_default_renderpass()
{
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

  VkAttachmentDescription attachments[2] = { color_attachment,
                                             depth_attachment };

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

  VkSubpassDependency dependencies[2] = { dependency, depth_dependency };

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

void
VulkanEngine::init_frame_buffers()
{
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
    VkImageView attachments[2] = { _swapchainImageViews[i], _depthImageView };
    fb_info.pAttachments = attachments;
    fb_info.attachmentCount = 2;
    VK_CHECK(
      vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));
  }
}

//==============================================================================
// Initialize Fences and Semaphores
//______________________________________________________________________________

void
VulkanEngine::init_sync_structures()
{
  VkFenceCreateInfo fenceCreateInfo = {
    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    .pNext = nullptr,
    .flags = VK_FENCE_CREATE_SIGNALED_BIT,
  };

  VkSemaphoreCreateInfo semaphoreCreateInfo = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
  };

  for (int i = 0; i < FRAME_OVERLAP; i++) {
    VK_CHECK(vkCreateFence(
      _device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

    VK_CHECK(vkCreateSemaphore(
      _device, &semaphoreCreateInfo, nullptr, &_frames[i]._presentSemaphore));

    VK_CHECK(vkCreateSemaphore(
      _device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));
  }
}

//==============================================================================
// Initialize Descriptor Sets
//______________________________________________________________________________

void
VulkanEngine::init_descriptors()
{
  std::vector<VkDescriptorPoolSize> sizes = {
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10 }
  };

  VkDescriptorPoolCreateInfo poolInfo = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .maxSets = 10,
    .poolSizeCount = (uint32_t)sizes.size(),
    .pPoolSizes = sizes.data(),
  };

  vkCreateDescriptorPool(_device, &poolInfo, nullptr, &_descriptorPool);

  // Setting up Scene Data using Dyanmic Descriptor Sets
  // Determine the size of the buffer by adding padding as needed
  const size_t sceneParamBufferSize =
    FRAME_OVERLAP * pad_uniform_buffer_size(sizeof(GPUSceneData));

  _sceneParameterBuffer = create_buffer(sceneParamBufferSize,
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VMA_MEMORY_USAGE_CPU_TO_GPU);

  // binding for camera data at 0
  VkDescriptorSetLayoutBinding cameraBind =
    vkinit::descriptorset_layout_binding(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);

  // binding for scene data at 1
  VkDescriptorSetLayoutBinding sceneBind = vkinit::descriptorset_layout_binding(
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    1);

  VkDescriptorSetLayoutBinding bindings[] = { cameraBind, sceneBind };

  VkDescriptorSetLayoutCreateInfo setInfo = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .bindingCount = 2,
    .pBindings = bindings,
  };

  vkCreateDescriptorSetLayout(_device, &setInfo, nullptr, &_globalSetLayout);

  for (int i = 0; i < FRAME_OVERLAP; i++) // Create buffers for each frame
  {

    _frames[i].cameraBuffer = create_buffer(sizeof(GPUCameraData),
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                            VMA_MEMORY_USAGE_CPU_TO_GPU);

    VkDescriptorSetAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = nullptr,
      .descriptorPool = _descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &_globalSetLayout,
    };

    vkAllocateDescriptorSets(_device, &allocInfo, &_frames[i].globalDescriptor);

    VkDescriptorBufferInfo cameraInfo = {
      .buffer = _frames[i].cameraBuffer._buffer,
      .offset = 0,
      .range = sizeof(GPUCameraData),
    };

    VkDescriptorBufferInfo sceneInfo = {
      .buffer = _sceneParameterBuffer._buffer,
      .offset = 0,
      .range = sizeof(GPUSceneData),
    };

    VkWriteDescriptorSet cameraWrite =
      vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                      _frames[i].globalDescriptor,
                                      &cameraInfo,
                                      0);

    VkWriteDescriptorSet sceneWrite =
      vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                      _frames[i].globalDescriptor,
                                      &sceneInfo,
                                      1);

    VkWriteDescriptorSet setWrites[] = { cameraWrite, sceneWrite };

    vkUpdateDescriptorSets(_device, 2, setWrites, 0, nullptr);
  }
}

//==============================================================================
// Initialize Pipelines
//______________________________________________________________________________

void
VulkanEngine::init_pipelines()
{
  // _________________________________________________________
  // Loading Shader Modules
  // ---------------------------------------------------------

  std::vector<std::pair<std::string, VkShaderModule*>> shaders = {
    { "shaders/triangle.vert.spv", &_triangleVertexShader },
    { "shaders/triangle2.vert.spv", &_triangle2VertexShader },
    { "shaders/triangle3.vert.spv", &_triangle3VertexShader },
    { "shaders/triangle.frag.spv", &_triangleFragShader },
    { "shaders/default_lit.frag.spv", &_colorMeshFragShader },
  };

  for (auto x : shaders) {
    auto filename{ x.first.c_str() };
    auto shader{ x.second };
    if (!load_shader_module(filename, shader)) {
      std::cerr << "[ERROR]: Unable to load: " << filename << std::endl;
    } else {
      std::cout << "Loaded: " << filename << std::endl;
    }
  }

  // _________________________________________________________
  //  Layout: Original
  // TODO: Remove dead code
  // ---------------------------------------------------------

  VkPipelineLayoutCreateInfo pipeline_layout_info =
    vkinit::pipeline_layout_create_info();

  VK_CHECK(vkCreatePipelineLayout(
    _device, &pipeline_layout_info, nullptr, &_trianglePipelineLayout));

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

  // push-constant setup
  mesh_pipeline_layout_info.pPushConstantRanges = &push_constant;
  mesh_pipeline_layout_info.pushConstantRangeCount = 1;

  // global-set layout
  mesh_pipeline_layout_info.setLayoutCount = 1;
  mesh_pipeline_layout_info.pSetLayouts = &_globalSetLayout;

  VK_CHECK(vkCreatePipelineLayout(
    _device, &mesh_pipeline_layout_info, nullptr, &_meshPipelineLayout));

  // _________________________________________________________
  // Pipeline Builder
  // TODO: Remove dead code
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

  pipelineBuilder._scissor.offset = { 0, 0 };
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
  pipelineBuilder._depthStencil =
    vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

  // // finally build the pipeline
  // _trianglePipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

  // _________________________________________________________
  //  Triangle 2 Pipeline!
  // TODO: Remove dead code
  // ---------------------------------------------------------

  // clear the shader stages for the builder
  pipelineBuilder._shaderStages.clear();

  // add the other shaders
  pipelineBuilder._shaderStages.push_back(
    vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT,
                                              _triangle2VertexShader));
  pipelineBuilder._shaderStages.push_back(
    vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT,
                                              _colorMeshFragShader));

  // build the triangle 2 pipeline
  // _triangle2Pipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

  // default depthtesting
  pipelineBuilder._depthStencil =
    vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

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
                                              _colorMeshFragShader));

  pipelineBuilder._pipelineLayout = _meshPipelineLayout;

  _meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

  create_material(_meshPipeline, _meshPipelineLayout, "defaultmesh");

  // _________________________________________________________
  // Free The Shader Modules
  // TODO: Use a loop if we end up needing more shaders
  // ---------------------------------------------------------
  vkDestroyShaderModule(_device, _triangleVertexShader, nullptr);
  vkDestroyShaderModule(_device, _triangleFragShader, nullptr);
  vkDestroyShaderModule(_device, _triangle2VertexShader, nullptr);
  vkDestroyShaderModule(_device, _colorMeshFragShader, nullptr);
  vkDestroyShaderModule(_device, _triangle3VertexShader, nullptr);
}

//==============================================================================
// Initialize Scene - converts the scene into renderable objects for the GPU
//______________________________________________________________________________

void
VulkanEngine::init_scene()
{
  RenderObject monkey = {
    .mesh = get_mesh("monkey"),
    .material = get_material("defaultmesh"),
    .transformMatrix = glm::mat4(1.0f),
  };

  _renderables.push_back(monkey);

  // creates N triangles that surround the monkey!
  int n = 10;
  for (int x = -n; x <= n; x++) {
    for (int y = -n; y <= n; y++) {
      for (int z = -n; z <= n; z++) {
        if (x == 0 || y == 0 || z == 0) {
          continue;
        }

        float X = x * 5 * (float(z) / n);
        float Y = y * 3 * (float(z) / n);
        float Z = z;

        glm::mat4 translation =
          glm::translate(glm::mat4{ 1.0 }, glm::vec3(X, Y, Z));

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
}

//==============================================================================
// Cleanup
//______________________________________________________________________________

void
VulkanEngine::cleanup()
{
  if (_isInitialized) {
    vkQueueWaitIdle(_graphicsQueue);

    vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
    for (auto x : _allocatedBuffers) {
      vmaDestroyBuffer(_allocator, x->_buffer, x->_allocation);
    }

    vkDestroyPipeline(_device, _meshPipeline, nullptr);
    vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
    vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);

    vkDestroyDescriptorSetLayout(_device, _globalSetLayout, nullptr);
    vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
    vmaDestroyBuffer(_allocator,
                     _sceneParameterBuffer._buffer,
                     _sceneParameterBuffer._allocation);

    for (int i = 0; i < FRAME_OVERLAP; i++) {
      vmaDestroyBuffer(_allocator,
                       _frames[i].cameraBuffer._buffer,
                       _frames[i].cameraBuffer._allocation);
      vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
      vkDestroySemaphore(_device, _frames[i]._presentSemaphore, nullptr);
      vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
      vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
    }

    vmaDestroyAllocator(_allocator);

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
// Draw (called every frame)
//______________________________________________________________________________

void
VulkanEngine::draw()
{
  // Wait until the GPU has finished rendering the last frame. Timeout 1 sec
  VK_CHECK(vkWaitForFences(
    _device, 1, &get_current_frame()._renderFence, true, 1000000000));

  VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

  // request image from the swapchain, one second timeout
  uint32_t swapchainImageIndex;
  VK_CHECK(vkAcquireNextImageKHR(_device,
                                 _swapchain,
                                 1000000000,
                                 get_current_frame()._presentSemaphore,
                                 nullptr,
                                 &swapchainImageIndex));

  // now that we are sure that the commands finished executing, we can safely
  // reset the command buffer to begin recording again.
  VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));

  // naming it cmd for shorter writing
  VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

  // begin the command buffer recording. We will use this command buffer
  // exactly once, so we want to let Vulkan know that
  VkCommandBufferBeginInfo cmdBeginInfo = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .pNext = nullptr,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    .pInheritanceInfo = nullptr,
  };

  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

  // This ClearValue will be used as the background on the screen
  VkClearValue clearValue;
  // float flash = abs(sin(_frameNumber / (5 * 60.f)));
  float flash = 0.5;
  clearValue.color = { { 0.0f, 0.0f, flash, 1.0f } };

  // clear depth at 1
  VkClearValue depthClear;
  depthClear.depthStencil.depth = 1.0f;

  VkClearValue clearValues[] = { clearValue, depthClear };

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
  rpInfo.renderArea.offset = { .x = 0, .y = 0 };
  rpInfo.renderArea.extent = _windowExtent;

  // finishes the rendering, and transitions to we image we specified, which
  // is "ready to be dislayed"
  VkDeviceSize offset = 0;

  vkCmdBindVertexBuffers(
    cmd, 0, 1, &_monkeyMesh._vertexBuffer._buffer, &offset);

  vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

  draw_objects(cmd, _renderables.data(), _renderables.size());

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
    .pWaitSemaphores = &get_current_frame()._presentSemaphore,
    .pWaitDstStageMask = &waitStage,
    .commandBufferCount = 1,
    .pCommandBuffers = &cmd,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &get_current_frame()._renderSemaphore,
  };

  // submit command buffer to the queue and execute it. _renderFence will now
  // block until the graphic commands finish execution
  VK_CHECK(vkQueueSubmit(
    _graphicsQueue, 1, &submit, get_current_frame()._renderFence));

  // -------------------------------------------------------------------------

  // this will put the image we just rendered into the visible window. we want
  // to wait on the _renderSemaphore for that, as it's necessary that drawing
  // commands have finished before the image is displayed to the user
  VkPresentInfoKHR presentInfo = {
    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    .pNext = nullptr,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &get_current_frame()._renderSemaphore,
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
// Run - enters into the main loop
//______________________________________________________________________________

void
VulkanEngine::run()
{
  SDL_Event e;
  bool bQuit = false;

  // begin frame timer
  auto t0 = std::chrono::high_resolution_clock::now();
  auto t1 = std::chrono::high_resolution_clock::now();
  float delta;

  // main loop
  while (!bQuit) {
    // calculate delta time
    t1 = std::chrono::high_resolution_clock::now();
    delta = std::chrono::duration<float>(t1 - t0).count();
    t0 = t1;

    // Handle events on queue
    while (SDL_PollEvent(&e) != 0) {
      // close the window when user alt-f4s or clicks the X button
      if (e.type == SDL_QUIT) {
        bQuit = true;
      } else if (e.type == SDL_KEYDOWN) {
        // nothing for the moment
      }
    }

    // handle keyboard movements
    const Uint8* state = SDL_GetKeyboardState(NULL);

    // FORWARD-BACKWARD
    if (state[SDL_SCANCODE_W]) {
      _playerPosition[2] += _playerSpeed * delta;
    }
    if (state[SDL_SCANCODE_S]) {
      _playerPosition[2] -= _playerSpeed * delta;
    }

    // LEFT-RIGHT
    if (state[SDL_SCANCODE_A]) {
      _playerPosition[0] += _playerSpeed * delta;
    }
    if (state[SDL_SCANCODE_D]) {
      _playerPosition[0] -= _playerSpeed * delta;
    }

    // UP-DOWN
    if (state[SDL_SCANCODE_Q]) {
      _playerPosition[1] -= _playerSpeed * delta;
    }
    if (state[SDL_SCANCODE_E]) {
      _playerPosition[1] += _playerSpeed * delta;
    }

    draw();
  }
}

//==============================================================================
// Load Shader Modules - handles opening and reading .SPV files
//______________________________________________________________________________

bool
VulkanEngine::load_shader_module(const char* filepath,
                                 VkShaderModule* outShaderModule)
{
  std::ifstream file(filepath, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "cannot open file: " << filepath << std::endl;
    return false;
  }
  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
  file.seekg(0);
  file.read((char*)buffer.data(), fileSize);
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
// Load Meshes - reads from .OBJ files, or generates them manually
//______________________________________________________________________________

void
VulkanEngine::load_meshes()
{
  _triangleMesh._vertices.resize(3);

  // vertex positions
  _triangleMesh._vertices[0].position = { 1.f, 1.f, 0.5f };
  _triangleMesh._vertices[1].position = { -1.f, 1.f, 0.5f };
  _triangleMesh._vertices[2].position = { 0.f, -1.f, 0.5f };

  // vertex colors, all green
  _triangleMesh._vertices[0].color = { 0.f, 1.f, 0.0f }; // pure green
  _triangleMesh._vertices[1].color = { 0.f, 1.f, 0.0f }; // pure green
  _triangleMesh._vertices[2].color = { 0.f, 1.f, 0.0f }; // pure green

  // monkey mesh!
  _monkeyMesh.load_from_obj("assets/monkey_smooth.obj");

  // sends the mesh data to the GPU
  upload_mesh(_triangleMesh);
  upload_mesh(_monkeyMesh);

  _meshes["monkey"] = _monkeyMesh;
  _meshes["triangle"] = _triangleMesh;
}

//==============================================================================
// Upload Mesh - writes mesh data to memory where the GPU can access it
//______________________________________________________________________________

void
VulkanEngine::upload_mesh(Mesh& mesh)
{
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

  VK_CHECK(vmaCreateBuffer(_allocator,
                           &bufferInfo,
                           &vmaallocInfo,
                           &mesh._vertexBuffer._buffer,
                           &mesh._vertexBuffer._allocation,
                           nullptr));

  // copy Vertex data
  void* data;
  vmaMapMemory(_allocator, mesh._vertexBuffer._allocation, &data);
  memcpy(data, mesh._vertices.data(), mesh._vertices.size() * sizeof(Vertex));
  vmaUnmapMemory(_allocator, mesh._vertexBuffer._allocation);
}

//==============================================================================
// Build Pipeline - Initializes a new graphics pipeline
//______________________________________________________________________________

VkPipeline
PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass)
{
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
  VkResult res = vkCreateGraphicsPipelines(
    device, VK_NULL_HANDLE, 1, &pipe, nullptr, &newPipeline);
  if (res != VK_SUCCESS) {
    std::cerr << "error: failed to create graphics pipeline" << std::endl;
    return VK_NULL_HANDLE;
  } else {
    return newPipeline;
  }
}

//==============================================================================
// Materials and Mesh Storage Management Functions
//______________________________________________________________________________

Material*
VulkanEngine::create_material(VkPipeline pipeline,
                              VkPipelineLayout layout,
                              const std::string& name)
{
  Material mat = {
    .pipeline = pipeline,
    .pipelineLayout = layout,
  };
  _materials[name] = mat;
  return &_materials[name];
}

// returns nullptr if not found
Material*
VulkanEngine::get_material(const std::string& name)
{
  auto it = _materials.find(name);
  if (it == _materials.end()) {
    return nullptr;
  } else {
    return &(*it).second;
  }
}

// returns nullptr if not found
Mesh*
VulkanEngine::get_mesh(const std::string& name)
{
  auto it = _meshes.find(name);
  if (it == _meshes.end()) {
    return nullptr;
  } else {
    return &(*it).second;
  }
}

//==============================================================================
// Draw Objects - tells the gpu to draw each object in the "first" array
//______________________________________________________________________________

void
VulkanEngine::draw_objects(VkCommandBuffer cmd, RenderObject* first, int count)
{
  Mesh* lastMesh = nullptr;

  Material* lastMaterial = nullptr;

  // calculate the current time. This should happen before looping, to ensure
  // that each object has the same time value.
  auto t = std::chrono::system_clock::now().time_since_epoch();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t);
  int time = -ms.count();

  // time is stored in the "data" field of push_constants
  glm::vec4 data = { glm::float32((time % 10000)), 0.0, 0.0, 0.0 };

  // make model view matrix for rendering the object (camera view/projection)
  // glm::vec3 camPos = {0.f, z_cam_pos, z_cam_pos};
  glm::vec3 camPos = _playerPosition;

  glm::mat4 view = glm::translate(glm::mat4(1.0f), camPos);

  glm::mat4 projection =
    glm::perspective(glm::radians(70.f), (1700.f / 900.f), 0.1f, 200.f);

  projection[1][1] *= -1;

  GPUCameraData camData = {
    .view = view,
    .proj = projection,
    .viewproj = projection * view,
  };

  // copies the camera data into a camera buffer, allowing us to bind it
  void* d;
  vmaMapMemory(_allocator, get_current_frame().cameraBuffer._allocation, &d);
  memcpy(d, &camData, sizeof(GPUCameraData));
  vmaUnmapMemory(_allocator, get_current_frame().cameraBuffer._allocation);

  // Loading the Scene Data, Lighting, etc. into buffer
  float framed = (_frameNumber / 120.f);
  _sceneParameters.ambientColor = { sin(framed), 0, cos(framed), 1 };
  char* sceneData;
  vmaMapMemory(
    _allocator, _sceneParameterBuffer._allocation, (void**)&sceneData);
  int frameIndex = _frameNumber % FRAME_OVERLAP;
  sceneData += pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;
  memcpy(sceneData, &_sceneParameters, sizeof(GPUSceneData));
  vmaUnmapMemory(_allocator, _sceneParameterBuffer._allocation);

  // loop for each of the renderable objects (e.g. each mesh or triangle)
  for (int i = 0; i < count; i++) {
    RenderObject& object = first[i];

    // The pipeline might already have our material bound.
    if (object.material != lastMaterial) {
      vkCmdBindPipeline(
        cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);

      lastMaterial = object.material;

      // offset for our scene buffer
      uint32_t uniform_offset =
        pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;

      vkCmdBindDescriptorSets(cmd,
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              object.material->pipelineLayout,
                              0,
                              1,
                              &get_current_frame().globalDescriptor,
                              1,
                              &uniform_offset);
    }

    MeshPushConstants constants{
      .data = data,
      .render_matrix = object.transformMatrix,
    };

    vkCmdPushConstants(cmd,
                       object.material->pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT,
                       0,
                       sizeof(MeshPushConstants),
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

//==============================================================================
// Get Current Frame (used for double buffering)
//______________________________________________________________________________

FrameData&
VulkanEngine::get_current_frame()
{
  return _frames[_frameNumber % FRAME_OVERLAP];
}

//==============================================================================
// Create Buffer
//______________________________________________________________________________

AllocatedBuffer
VulkanEngine::create_buffer(size_t allocSize,
                            VkBufferUsageFlags usage,
                            VmaMemoryUsage memoryUsage)
{
  VkBufferCreateInfo bufferInfo = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .pNext = nullptr,
    .size = allocSize,
    .usage = usage,
  };

  VmaAllocationCreateInfo vmaAllocInfo = {
    .usage = memoryUsage,
  };

  AllocatedBuffer newBuffer;

  VK_CHECK(vmaCreateBuffer(_allocator,
                           &bufferInfo,
                           &vmaAllocInfo,
                           &newBuffer._buffer,
                           &newBuffer._allocation,
                           nullptr));

  return newBuffer;
}

size_t
VulkanEngine::pad_uniform_buffer_size(size_t originalSize)
{
  // Calculate required alignment based on minimum device offset alignment
  size_t minUboAlignment =
    _gpuProperties.limits.minUniformBufferOffsetAlignment;
  size_t alignedSize = originalSize;
  if (minUboAlignment > 0) {
    alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
  }
  return alignedSize;
}