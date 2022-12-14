#include <vk_initializers.h>

VkCommandPoolCreateInfo
vkinit::command_pool_create_info(uint32_t queueFamilyIndex,
                                 VkCommandPoolCreateFlags flags /*= 0*/
)
{
  VkCommandPoolCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.pNext = nullptr;

  info.queueFamilyIndex = queueFamilyIndex;
  info.flags = flags;
  return info;
}

VkCommandBufferAllocateInfo
vkinit::command_buffer_allocate_info(
  VkCommandPool pool,
  uint32_t count /*= 1*/,
  VkCommandBufferLevel level /*= VK_COMMAND_BUFFER_LEVEL_PRIMARY*/
)
{
  VkCommandBufferAllocateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  info.pNext = nullptr;

  info.commandPool = pool;
  info.commandBufferCount = count;
  info.level = level;
  return info;
}

VkPipelineShaderStageCreateInfo
vkinit::pipeline_shader_stage_create_info(VkShaderStageFlagBits stage,
                                          VkShaderModule shaderModule)
{
  VkPipelineShaderStageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.pNext = nullptr;
  info.stage = stage;
  info.module = shaderModule;
  info.pName = "main";
  return info;
}

VkPipelineVertexInputStateCreateInfo
vkinit::vertex_input_create_info()
{
  VkPipelineVertexInputStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.vertexBindingDescriptionCount = 0;
  info.vertexAttributeDescriptionCount = 0;
  return info;
}

VkPipelineInputAssemblyStateCreateInfo
vkinit::input_assembly_create_info(VkPrimitiveTopology topology)
{
  VkPipelineInputAssemblyStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.topology = topology;
  info.primitiveRestartEnable = VK_FALSE;
  return info;
}

VkPipelineRasterizationStateCreateInfo
vkinit::rasterization_state_create_info(VkPolygonMode polygonMode)
{
  VkPipelineRasterizationStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.depthClampEnable = VK_FALSE;
  info.rasterizerDiscardEnable = VK_FALSE;
  info.polygonMode = polygonMode;
  info.lineWidth = 1.0f;

  // no backface culling
  info.cullMode = VK_CULL_MODE_NONE;
  info.frontFace = VK_FRONT_FACE_CLOCKWISE;

  // no depth bias
  info.depthBiasEnable = VK_FALSE;
  info.depthBiasConstantFactor = 0.0f;
  info.depthBiasClamp = 0.0f;
  info.depthBiasSlopeFactor = 0.0f;

  return info;
}

VkPipelineMultisampleStateCreateInfo
vkinit::multisampling_state_create_info()
{
  VkPipelineMultisampleStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.sampleShadingEnable = VK_FALSE;
  info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  info.minSampleShading = 1.0f;
  info.pSampleMask = nullptr;
  info.alphaToCoverageEnable = VK_FALSE;
  info.alphaToOneEnable = VK_FALSE;
  return info;
}

VkPipelineColorBlendAttachmentState
vkinit::color_blend_attachment_state()
{
  VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
  colorBlendAttachment.colorWriteMask =
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;
  return colorBlendAttachment;
}

VkPipelineLayoutCreateInfo
vkinit::pipeline_layout_create_info()
{
  VkPipelineLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  info.pNext = nullptr;

  // empty defaults
  info.flags = 0;
  info.setLayoutCount = 0;
  info.pSetLayouts = nullptr;
  info.pushConstantRangeCount = 0;
  info.pPushConstantRanges = nullptr;
  return info;
}

VkImageCreateInfo
vkinit::image_create_info(VkFormat format,
                          VkImageUsageFlags usageFlags,
                          VkExtent3D extent)
{
  return VkImageCreateInfo{
    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .pNext = nullptr,
    .imageType = VK_IMAGE_TYPE_2D,
    .format = format,
    .extent = extent,
    .mipLevels = 1,
    .arrayLayers = 1,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .tiling = VK_IMAGE_TILING_OPTIMAL,
    .usage = usageFlags,
  };
}

VkImageViewCreateInfo
vkinit::imageview_create_info(VkFormat format,
                              VkImage image,
                              VkImageAspectFlags aspectFlags)
{
  VkImageViewCreateInfo info{
    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    .pNext = nullptr,
    .image = image,
    .viewType = VK_IMAGE_VIEW_TYPE_2D,
    .format = format,
  };
  info.subresourceRange.baseMipLevel = 0, info.subresourceRange.levelCount = 1;
  info.subresourceRange.baseArrayLayer = 0;
  info.subresourceRange.layerCount = 1;
  info.subresourceRange.aspectMask = aspectFlags;
  return info;
}

VkPipelineDepthStencilStateCreateInfo
vkinit::depth_stencil_create_info(bool bDepthTest,
                                  bool bDepthWrite,
                                  VkCompareOp compareOp)
{
  return VkPipelineDepthStencilStateCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
    .pNext = nullptr,
    .depthTestEnable = bDepthTest ? VK_TRUE : VK_FALSE,
    .depthWriteEnable = bDepthWrite ? VK_TRUE : VK_FALSE,
    .depthCompareOp = bDepthTest ? compareOp : VK_COMPARE_OP_ALWAYS,
    .depthBoundsTestEnable = VK_FALSE,
    .stencilTestEnable = VK_FALSE,
    .minDepthBounds = 0.0f, // Optional
    .maxDepthBounds = 1.0f, // Optional
  };
}

VkDescriptorSetLayoutBinding
vkinit::descriptorset_layout_binding(VkDescriptorType type,
                                     VkShaderStageFlags stageFlags,
                                     uint32_t binding)
{
  VkDescriptorSetLayoutBinding setbind = {};
  setbind.binding = binding;
  setbind.descriptorCount = 1;
  setbind.descriptorType = type;
  setbind.pImmutableSamplers = nullptr;
  setbind.stageFlags = stageFlags;

  return setbind;
}

VkWriteDescriptorSet
vkinit::write_descriptor_buffer(VkDescriptorType type,
                                VkDescriptorSet dstSet,
                                VkDescriptorBufferInfo* bufferInfo,
                                uint32_t binding)
{
  VkWriteDescriptorSet write = {};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.pNext = nullptr;
  write.dstBinding = binding;
  write.dstSet = dstSet;
  write.descriptorCount = 1;
  write.descriptorType = type;
  write.pBufferInfo = bufferInfo;

  return write;
}
