// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

namespace vkinit {

VkCommandPoolCreateInfo
command_pool_create_info(uint32_t queueFamilyIndex,
                         VkCommandPoolCreateFlags flags = 0);

VkCommandBufferAllocateInfo command_buffer_allocate_info(
    VkCommandPool pool, uint32_t count = 1,
    VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

// creates information about a single shader stage for the pipeline. We will
// build it from a shader stage and a shader module.
VkPipelineShaderStageCreateInfo
pipeline_shader_stage_create_info(VkShaderStageFlagBits stage,
                                  VkShaderModule shaderModule);

// creates info for vertex buffers and vertex formats.
VkPipelineVertexInputStateCreateInfo vertex_input_create_info();

// configuration for the 'topology' and primitive shapes that will be draw
// (e.g. triangles / points / lines)
VkPipelineInputAssemblyStateCreateInfo
input_assembly_create_info(VkPrimitiveTopology topology);

// configuration for the fixed-function rasterization stage
VkPipelineRasterizationStateCreateInfo
rasterization_state_create_info(VkPolygonMode polygonMode);

VkPipelineMultisampleStateCreateInfo multisampling_state_create_info();

VkPipelineColorBlendAttachmentState color_blend_attachment_state();

VkPipelineLayoutCreateInfo pipeline_layout_create_info();

VkImageCreateInfo image_create_info(VkFormat format,
                                            VkImageUsageFlags usageFlags,
                                            VkExtent3D extent);

VkImageViewCreateInfo imageview_create_info(VkFormat format, VkImage image,
                                            VkImageAspectFlags aspectFlags);

VkPipelineDepthStencilStateCreateInfo
depth_stencil_create_info(bool bDepthTest, bool bDepthWrite,
                          VkCompareOp compareOp);

} // namespace vkinit
