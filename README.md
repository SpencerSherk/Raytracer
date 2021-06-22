Ray Tracing: Triangle Meshes and AABB Trees
===========================================

This raytracer works for SDF primitives, and triangle meshes (.OFF and .OBJ). 

It uses an AABB acceleration structure for a 500X speedup in computation/render time.
This acceleration structure uses a bottom up approach, merging nodes in the bottom level first, and then working up to the root node of the tree.
