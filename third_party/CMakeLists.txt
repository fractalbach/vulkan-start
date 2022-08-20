find_package(Vulkan REQUIRED)

# vkbootstrap
add_library(vkbootstrap STATIC)
target_sources(vkbootstrap PRIVATE 
    vkbootstrap/VkBootstrap.h
    vkbootstrap/VkBootstrap.cpp
)
target_include_directories(vkbootstrap PUBLIC vkbootstrap)
target_link_libraries(vkbootstrap PUBLIC Vulkan::Vulkan $<$<BOOL:UNIX>:${CMAKE_DL_LIBS}>)

# vma
# (header-only lib)
add_library(vma INTERFACE)
target_include_directories(vma INTERFACE vma)

# glm 
# (header-only lib)
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE glm)

# tinyobjloader
add_library(tinyobjloader STATIC)
target_sources(tinyobjloader PRIVATE
    tinyobjloader/tiny_obj_loader.h
    tinyobjloader/tiny_obj_loader.cc
)
target_include_directories(tinyobjloader PUBLIC tinyobjloader)


# # SDL2 from git submodule
set(SDL_STATIC ON)
set(SDL_SHARED OFF)
add_subdirectory(SDL2 EXCLUDE_FROM_ALL)
add_library(SDL2 ALIAS SDL2-static)


# imgui
add_library(imgui STATIC)
target_include_directories(imgui PUBLIC imgui)
target_sources(imgui PRIVATE 
    imgui/imgui.h
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_impl_vulkan.cpp
    imgui/imgui_impl_sdl.cpp
)

# stb_image
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE stb_image)
 

# Link libraries
target_link_libraries(imgui PUBLIC Vulkan::Vulkan SDL2-static)
