#include <iostream>
#include <tiny_obj_loader.h>
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

// =============================================================================
bool Mesh::load_from_obj(const char *filename) {

  tinyobj::attrib_t attrib; // contains vertex arrays
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn;
  std::string err;

  tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename,
                   nullptr);

  if (!warn.empty()) {
    std::cerr << "[WARNING]: " << warn << std::endl;
  }

  if (!err.empty()) {
    std::cerr << "[ERROR]: " << err << std::endl;
    return false;
  }

  // ---------------------------------------------------------------------------
  // Put meshes from file into the vertex buffer

  // Loop over shapes
  for (size_t s = 0; s < shapes.size(); s++) {

    // Loop over polygon faces
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

      int fv = 3; // hardcoded vertices per face

      for (size_t v = 0; v < fv; v++) {

        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
        tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
        tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
        tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

        // copy into a new vertex object
        Vertex vert;
        vert.position.x = vx;
        vert.position.y = vy;
        vert.position.z = vz;
        vert.normal.x = nx;
        vert.normal.y = ny;
        vert.normal.z = nz;
        vert.color = vert.normal;

        _vertices.push_back(vert);
      }
      index_offset += fv;
    }
  }

  return true;
}