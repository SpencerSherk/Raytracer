Ray Tracing: Triangle Meshes and AABB Trees
===========================================

Raytracer works for SDF primitives, and triangle meshes (.OFF and .OBJ). 

Uses AABB acceleration structures to make the computation faster.
To construct the AABB tree, I used a bottom up approach- merging the nodes in the bottom level first, and then working my way up to the root node of the tree.
