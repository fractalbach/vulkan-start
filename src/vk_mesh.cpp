#include <vk_mesh.h>

VertexInputDescription Vertex::get_vertex_description() {

  VkVertexInputBindingDescription mainBinding = {
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
  };

  // LOCATION 0 : POSITION
  VkVertexInputAttributeDescription positionAttribute = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32B32_SFLOAT,
      .offset = offsetof(Vertex, position),
  };

  // LOCATION 1 : NORMAL
  VkVertexInputAttributeDescription normalAttribute = {
      .location = 1,
      .binding = 0,
      .format = VK_FORMAT_R32G32B32_SFLOAT,
      .offset = offsetof(Vertex, normal),
  };

  // LOCATION 2 : COLOR
  VkVertexInputAttributeDescription colorAttribute = {
      .location = 2,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
      .offset = offsetof(Vertex, color),
  };

  VertexInputDescription description = {
      .bindings = {mainBinding},
      .attributes = {positionAttribute, normalAttribute, colorAttribute},
  };

  return description;
}