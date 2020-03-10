////////////////////////////////////////////////////////////////////////////////
// C++ include
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <stack>

// Eigen for matrix operations
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION 
#include "stb_image_write.h"
#include "utils.h"

// JSON parser library (https://github.com/nlohmann/json)
#include "json.hpp"
using json = nlohmann::json;

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
// Define types & classes
////////////////////////////////////////////////////////////////////////////////

struct Ray {
    Vector3d origin;
    Vector3d direction;
    Ray() { }
    Ray(Vector3d o, Vector3d d) : origin(o), direction(d) { }
};

struct Light {
    Vector3d position;
    Vector3d intensity;
};

struct Intersection {
    Vector3d position;
    Vector3d normal;
    double ray_param;
};

struct Camera {
    bool is_perspective;
    Vector3d position;
    double field_of_view; // between 0 and PI
    double focal_length;
    double lens_radius; // for depth of field
};

struct Material {
    Vector3d ambient_color;
    Vector3d diffuse_color;
    Vector3d specular_color;
    double specular_exponent; 

    Vector3d reflection_color;
    Vector3d refraction_color;
    double refraction_index;
};

struct Object {
    Material material;
    virtual ~Object() = default; // Classes with virtual methods should have a virtual destructor!
    virtual bool intersect(const Ray &ray, Intersection &hit) = 0;
};

// We use smart pointers to hold objects as this is a virtual class
typedef std::shared_ptr<Object> ObjectPtr;

struct Sphere : public Object {
    Vector3d position;
    double radius;

    virtual ~Sphere() = default;
    virtual bool intersect(const Ray &ray, Intersection &hit) override;
};

struct Parallelogram : public Object {
    Vector3d origin;
    Vector3d u;
    Vector3d v;

    virtual ~Parallelogram() = default;
    virtual bool intersect(const Ray &ray, Intersection &hit) override;
};

struct AABBTree {
    struct Node {
        AlignedBox3d bbox;
        int parent; // Index of the parent node (-1 for root)
        int left; // Index of the left child (-1 for a leaf)
        int right; // Index of the right child (-1 for a leaf)
        int triangle; // Index of the node triangle (-1 for internal nodes)
        int index; // ADDED: Index in the nodes vector for the AABB tree
    };

    struct Triangle {
        Vector3d a;
        Vector3d b;
        Vector3d c;
    };
    std::vector<Node> nodes;
    std::vector<Triangle> triangles;
    int root;

    AABBTree() = default; // Default empty constructor
    AABBTree(const MatrixXd &V, const MatrixXi &F); // Build a BVH from an existing mesh
};

struct Mesh : public Object {
    MatrixXd vertices; // n x 3 matrix (n points)
    MatrixXi facets; // m x 3 matrix (m triangles)

    AABBTree bvh;

    Mesh() = default; // Default empty constructor
    Mesh(const std::string &filename);
    virtual ~Mesh() = default;
    virtual bool intersect(const Ray &ray, Intersection &hit) override;
};

struct Scene {
    Vector3d background_color;
    Vector3d ambient_light;

    Camera camera;
    std::vector<Material> materials;
    std::vector<Light> lights;
    std::vector<ObjectPtr> objects;
};

////////////////////////////////////////////////////////////////////////////////

// Read a triangle mesh from an off file
void load_off(const std::string &filename, MatrixXd &V, MatrixXi &F) {
    std::ifstream in(filename);
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    V.resize(nv, 3);
    F.resize(nf, 3);
    for (int i = 0; i < nv; ++i) {
        in >> V(i, 0) >> V(i, 1) >> V(i, 2);
    }
    for (int i = 0; i < nf; ++i) {
        int s;
        in >> s >> F(i, 0) >> F(i, 1) >> F(i, 2);
        assert(s == 3);
    }
}

Mesh::Mesh(const std::string &filename) {
    // Load a mesh from a file (assuming this is a .off file), and create a bvh
    load_off(filename, vertices, facets);
    bvh = AABBTree(vertices, facets);
}

////////////////////////////////////////////////////////////////////////////////
// BVH Implementation
////////////////////////////////////////////////////////////////////////////////

// Bounding box of a triangle
AlignedBox3d bbox_triangle(const Vector3d &a, const Vector3d &b, const Vector3d &c) {
    AlignedBox3d box;
    box.extend(a);
    box.extend(b);
    box.extend(c);
    return box;
}

// helper function for cost function. calculates increased volume of merged bounding box
double getAdditionalVol(AlignedBox3d &box1, AlignedBox3d &box2) {
    double vol1 = box1.volume();
    double vol2 = box2.volume();
    double additionalVolume = box1.merged(box2).volume() - (vol1 + vol2);
    return additionalVolume;
}

AABBTree::AABBTree(const MatrixXd &V, const MatrixXi &F) {

    // Compute the centroids of all the triangles in the input mesh
    MatrixXd centroids(F.rows(), V.cols());

    centroids.setZero();
    for (int i = 0; i < F.rows(); ++i) {
        for (int k = 0; k < F.cols(); ++k) {
            centroids.row(i) += V.row(F(i, k));
        }
        centroids.row(i) /= F.cols();
    }

    // Bottom-up approach.
    // Merge nodes 2 by 2, starting from the leaves of the forest, until only 1 tree is left.

    AABBTree t;
    std::vector<Node> list;
    std::vector<Node> mergedList;

    // Create a leaf node for every input triangle: N1, N2, ... Nk. Let S = {N1, N2, ..., Nk}.
    for(int i = 0; i < F.rows(); ++i) {
        AABBTree::Node n;
        Vector3d a = V.row(F(i,0));
        Vector3d b = V.row(F(i,1));
        Vector3d c = V.row(F(i,2));
        AlignedBox3d box = bbox_triangle(a,b,c);
        AABBTree::Triangle nTri;
        nTri.a = a; nTri.b = b; nTri.c = c;
        triangles.push_back(nTri);
        n.bbox = box;
        n.triangle = i;
        n.left = -1;
        n.right = -1;
        n.index = i;
        t.nodes.push_back(n); 
        list.push_back(n);    
    }

    // for storing index in nodes vector for each merged node
    int mergedIndex = nodes.size();

    // merge nodes level by level until we have only the root node remaining
    do {
        // Begin merging next level, move merged list into active list
        if (mergedList.size() > 1) {
            assert(list.size == 0);
            for (int i = 0; i < list.size(); i++) {
                list.push_back(mergedList.at(i));
            }
            mergedList.clear(); // empty merged list
        }

        // merge nodes in active list, loop until all nodes at this level are merged
        while (list.size() > 1) {
            AABBTree::Node curr = list[0];
            AABBTree::Node bestMatch;
            double minAdditionalVol = 100000; // arbitrary large value
            int bmIndex; // index of bestMatch in nodes list

            // find best match for current node
            for (int i = 1; i < list.size(); i++) {
                double newVol = getAdditionalVol(curr.bbox, list[i].bbox);
                if (newVol < minAdditionalVol) {
                    minAdditionalVol = newVol;
                    bestMatch = list[i];
                    bmIndex = i;
                } 
            }

            // set variables for merged node
            AABBTree::Node merged;
            merged.bbox = curr.bbox.merged(bestMatch.bbox);
            merged.index = mergedIndex;
            merged.left = -1;
            merged.right = -1;

            // set parent field for curr and best match to equal merged index
            curr.parent = mergedIndex;
            bestMatch.parent = mergedIndex;
            mergedIndex ++;

            // remove merged nodes from list 
            list.erase(list.begin() + curr.index);
            list.erase(list.begin() + bestMatch.index);
            mergedList.push_back(merged);

        }
    } while (mergedList.size() > 1); // keep looping while we are not at root
}

////////////////////////////////////////////////////////////////////////////////

bool Sphere::intersect(const Ray &ray, Intersection &hit) {
    Vector3d ray_origin = ray.orign;
    Vector3d ray_direction = RowVector3d(0,0,-1);
    return false;
}

bool Parallelogram::intersect(const Ray &ray, Intersection &hit) {
        MatrixXd A(3,3);
        for (unsigned i=0;i<3;i++) {
            for (unsigned j=0;j<3;j++) {
                if (j==0) {
                    A(i,j) = u(i);
                }
                if (j==1){
                    A(i,j) = v(i);
                }
                if (j==2){
                    A(i,j) = ray.direction(i);         
                }
            }
        }

        // populate the vector b
        Vector3d b;
        for (unsigned i=0;i<3;i++) {
            b(i) = origin(i) - ray.origin(i);
        }

        // solve linear eq Ax=b for x, return this vector 
        Vector3d x = A.colPivHouseholderQr().solve(b);

    if ((x(2) > 0) && (0 <= x(0)) && (0 <= x(1)) && (x(0)+x(1) <= 2) && (x(1) <= 1) && (x(0) <= 1))
        return true;
    else 
        return false;
}

// -----------------------------------------------------------------------------


bool intersect_triangle(const Ray &ray, const Vector3d &a, const Vector3d &b, const Vector3d &c, Intersection &hit) {

    // Compute whether the ray intersects the given triangle. 
    MatrixXd A(3,3);
    for (unsigned i=0;i<3;i++) { // cols
        for (unsigned j=0;j<3;j++) { // rows
            if (j==0) {
                A(i,j) = a(i)-b(i);
            }
            if (j==1){
                A(i,j) = a(i)-c(i);
            }
            if (j==2){
                A(i,j) = ray.direction(i);
            }
        }
    }

    // populate the vector b
    Vector3d b_new;
    for (unsigned i=0;i<3;i++) {
        b_new(i) = a(i) - ray.origin(i);
    }

    // solve linear eq Ax=b for x, return this vector
    Vector3d x = A.colPivHouseholderQr().solve(b_new);

    if (x(2)>0 && 0 <= x(0) && 0 <= x(1) && x(0)+x(1) <= 1){
        hit.position = ray.origin+x(2)*ray.direction;
        hit.normal= (b-a).cross(c-a).normalized();
        return true;
    }

    return false;
}


// AABB bounding box intersection function
bool intersect_box(const Ray &ray, const AlignedBox3d &box) {

    /*
       -- construct the bounding box 
       solve for t (ex: e2+d2.dot(t) = zmax)
       if in range of x min and x max then there is an intersection 
       (if intersecting one of x y or z, then break)
       check intersection with 6 planes derived from box.min and box.max 
       for each three directions using (e+dt), check intersect
       Compute whether the ray intersects the given box.
       There is no need to set the resulting normal and ray parameter, since
       we are not testing with the real surface here anyway.  
       - using intersection algo described on scratchapixel.com
    */

    // X 
    double tmin = (box.min()(0) - ray.origin(0) / ray.direction(0)); 
    double tmax = (box.max()(0) - ray.origin(0) / ray.direction(0)); 
    if (tmin > tmax) std::swap(tmin, tmax); 

    // Y
    double tymin = (box.min()(1) - ray.origin(1)) / ray.direction(1); 
    double tymax = (box.max()(1) - ray.origin(1)) / ray.direction(1); 
    if (tymin > tymax) std::swap(tymin, tymax); 
    if ((tmin > tymax) || (tymin > tmax)) 
        return false; 
    if (tymin > tmin) 
        tmin = tymin; 
    if (tymax < tmax) 
        tmax = tymax; 

    // Z
    double tzmin = (box.min()(2) - ray.origin(2)) / ray.direction(2); 
    double tzmax = (box.max()(2) - ray.origin(2)) / ray.direction(2); 
    if (tzmin > tzmax) std::swap(tzmin, tzmax); 
    if ((tmin > tzmax) || (tzmin > tmax)) 
        return false; 
    if (tzmin > tmin) 
        tmin = tzmin; 
    if (tzmax < tmax) 
        tmax = tzmax; 

    return true; 
}

bool Mesh::intersect(const Ray &ray, Intersection &closest_hit) {
    // Method (1): Traverse every triangle and return the closest hit. . . . .
    // for every row in F, build triangle with v.row(f(r,0)) ...1...2 --> then input this vector to the ray-triangle intersect
    for (int i = 0; i < facets.rows(); ++i) {
        Vector3d a = vertices.row(facets(i,0));
        Vector3d b = vertices.row(facets(i,1));
        Vector3d c = vertices.row(facets(i,2));

        if (intersect_triangle(ray, a, b, c, closest_hit) == true) {  
            return true;   
        }
    }
    
    return false;
}


////////////////////////////////////////////////////////////////////////////////
// Define ray-tracing functions
////////////////////////////////////////////////////////////////////////////////

// Function declaration here (could be put in a header file)
Vector3d ray_color(const Scene &scene, const Ray &ray, const Object &object, const Intersection &hit, int max_bounce);
Object * find_nearest_object(const Scene &scene, const Ray &ray, Intersection &closest_hit);
Vector3d shoot_ray(const Scene &scene, const Ray &ray, int max_bounce);

// -----------------------------------------------------------------------------

Vector3d ray_color(const Scene &scene, const Ray &ray, const Object &obj, const Intersection &hit, int max_bounce) {
    
    // Material for hit object
    const Material &mat = obj.material;

    // Ambient light contribution
    Vector3d ambient_color = obj.material.ambient_color.array() * scene.ambient_light.array();

    // Punctual lights contribution (direct lighting)
    Vector3d lights_color(0, 0, 0);

    for (const Light &light : scene.lights) {

        Vector3d Li = (light.position - hit.position).normalized();
        Vector3d N = hit.normal;

        // Shoot a shadow ray to determine if the light should affect the intersection point
        // because hit is passed in as a constant, we make a local clone
        Intersection shadow_hit = hit;
        Ray shadow_ray;
        shadow_ray.direction = light.position - hit.position;
        shadow_ray.origin = hit.position;

        // offset shadow origin by small epsilon value to avoid shadow noise
        for (int i=0; i<3; i++ ){

            if (shadow_ray.direction(i) > 0)
                shadow_ray.origin(i) += 0.00001;
            else 
                shadow_ray.origin(i) -= 0.00001;
        }

        // test if current point is inside shadow
        bool insideShadow = false;

        for (int i=0; i<scene.objects.size(); i++) {

            if (scene.objects[i]->intersect(shadow_ray, shadow_hit)) {
                insideShadow = true;
                break; // exit loop if object between light and current point (if shadow)
            }
        } 

        // Diffuse contribution
        Vector3d diffuse = mat.diffuse_color * std::max(Li.dot(N), 0.0);

        // Specular contribution
        double specular_factor = ((light.position-hit.position).normalized().transpose() * N);
        Vector3d specular(specular_factor, specular_factor, specular_factor);

        // Attenuate lights according to the squared distance to the lights
        Vector3d D = light.position - hit.position;
        lights_color += (diffuse + specular).cwiseProduct(light.intensity) /  D.squaredNorm();
    }

    // Compute the color of the reflected ray and add its contribution to the current point color.
    // calculate angle of reflection ray
    Ray reflection_ray;

    // calculate direction of reflection ray
    reflection_ray.direction = ray.direction - 2 * ray.direction.dot(hit.normal) * hit.normal;
    reflection_ray.origin = hit.position;
    Vector3d reflection_color(0, 0, 0);

    // shoot reflection ray to calculate color
    reflection_color = shoot_ray(scene, reflection_ray, max_bounce-1); 

    Vector3d D = light.position - hit.position;

    // apply shadows
    if (!insideShadow)
        lights_color += (diffuse + specular).cwiseProduct(light.intensity) /  D.squaredNorm();

    // Rendering equation
    Vector3d C = ambient_color + lights_color + reflection_color + refraction_color;

    return C;
}

// -----------------------------------------------------------------------------

Object * find_nearest_object(const Scene &scene, const Ray &ray, Intersection &closest_hit) {
    
    int closest_index = -1;

    // init lowest dist to arbitrarily huge number
    double lowest_dist = (100000000000);

    for (int i=0; i<scene.objects.size(); i++){
        if (scene.objects[i]->intersect(ray, closest_hit)) {
            Vector3d diff = ray.origin - closest_hit.position;

            // for each intersection point, calculate distance from camera, and keep track of smallest dist
            double dist = sqrt(diff(0) * diff(0) + diff(1) * diff(1) + diff(2) * diff(2));
            if (dist < lowest_dist) {
                lowest_dist = dist;
                closest_index = i;
            }
        }
    }
    if (closest_index < 0) {
        // Return a NULL pointer
        return nullptr;
    } else {
        // Return a pointer to the hit object. Don't forget to set 'closest_hit' accordingly!
        return scene.objects[closest_index].get();
    }

    if (closest_index < 0) {
        // Return a NULL pointer
        return nullptr;
    } else {
        // Return a pointer to the hit object. Don't forget to set 'closest_hit' accordingly!
        return scene.objects[closest_index].get();
    }
}

Vector3d shoot_ray(const Scene &scene, const Ray &ray, int max_bounce) {
    Intersection hit;
    if (Object * obj = find_nearest_object(scene, ray, hit)) {
        // printf("\nfound nearest object\n");
        // 'obj' is not null and points to the object of the scene hit by the ray
        return ray_color(scene, ray, *obj, hit, max_bounce);
    } else {
        // 'obj' is null, we must return the background color
        return scene.background_color;
    }
}

////////////////////////////////////////////////////////////////////////////////

void render_scene(const Scene &scene) {
    std::cout << "Simple ray tracer..." << std::endl;

    int w = 640;
    int h = 480;
    MatrixXd R = MatrixXd::Zero(w, h);
    MatrixXd G = MatrixXd::Zero(w, h);
    MatrixXd B = MatrixXd::Zero(w, h);
    MatrixXd A = MatrixXd::Zero(w, h); // Store the alpha mask

    // The camera always points in the direction -z
    // The sensor grid is at a distance 'focal_length' from the camera center,
    // and covers an viewing angle given by 'field_of_view'.
    double aspect_ratio = double(w) / double(h);
    double scale_y = 1.0; // TODO: Stretch the pixel grid by the proper amount here
    double scale_x = 1.0; //

    // The pixel grid through which we shoot rays is at a distance 'focal_length'
    // from the sensor, and is scaled from the canonical [-1,1] in order
    // to produce the target field of view.
    Vector3d grid_origin(-scale_x, scale_y, -scene.camera.focal_length);
    Vector3d x_displacement(2.0/w*scale_x, 0, 0);
    Vector3d y_displacement(0, -2.0/h*scale_y, 0);

    for (unsigned i = 0; i < w; ++i) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Ray tracing: " << (100.0 * i) / w << "%\r" << std::flush;
        for (unsigned j = 0; j < h; ++j) {
            Vector3d shift = grid_origin + (i+0.5)*x_displacement + (j+0.5)*y_displacement;

            // Prepare the ray
            Ray ray;

            if (scene.camera.is_perspective) {
                // Perspective camera
                Vector3d P = scene.camera.position + Vector3d(shift[0], shift[1], 0);
                Vector3d E = P + Vector3d(0,0,5);
                ray.direction = (P - E).normalized();
                ray.origin = E;
            } else {
                // Orthographic camera
                ray.origin = scene.camera.position + Vector3d(shift[0], shift[1], 0);
                ray.direction = Vector3d(0, 0, -1);
            }

            int max_bounce = 5;
            Vector3d C = shoot_ray(scene, ray, max_bounce);
            R(i, j) = C(0);
            G(i, j) = C(1);
            B(i, j) = C(2);
            A(i, j) = 1;
        }
    }

    std::cout << "Ray tracing: 100%  " << std::endl;

    // Save to png
    const std::string filename("raytrace.png");
    write_matrix_to_png(R, G, B, A, filename);
}

////////////////////////////////////////////////////////////////////////////////

Scene load_scene(const std::string &filename) {
    Scene scene;

    // Load json data from scene file
    json data;
    std::ifstream in(filename);
    in >> data;

    // Helper function to read a Vector3d from a json array
    auto read_vec3 = [] (const json &x) {
        return Vector3d(x[0], x[1], x[2]);
    };

    // Read scene info
    scene.background_color = read_vec3(data["Scene"]["Background"]);
    scene.ambient_light = read_vec3(data["Scene"]["Ambient"]);

    // Read camera info
    scene.camera.is_perspective = data["Camera"]["IsPerspective"];
    scene.camera.position = read_vec3(data["Camera"]["Position"]);
    scene.camera.field_of_view = data["Camera"]["FieldOfView"];
    scene.camera.focal_length = data["Camera"]["FocalLength"];
    scene.camera.lens_radius = data["Camera"]["LensRadius"];

    // Read materials
    for (const auto &entry : data["Materials"]) {
        Material mat;
        mat.ambient_color = read_vec3(entry["Ambient"]);
        mat.diffuse_color = read_vec3(entry["Diffuse"]);
        mat.specular_color = read_vec3(entry["Specular"]);
        mat.reflection_color = read_vec3(entry["Mirror"]);
        mat.refraction_color = read_vec3(entry["Refraction"]);
        mat.refraction_index = entry["RefractionIndex"];
        mat.specular_exponent = entry["Shininess"];
        scene.materials.push_back(mat);
    }

    // Read lights
    for (const auto &entry : data["Lights"]) {
        Light light;
        light.position = read_vec3(entry["Position"]);
        light.intensity = read_vec3(entry["Color"]);
        scene.lights.push_back(light);
    }

    // Read objects
    for (const auto &entry : data["Objects"]) {
        ObjectPtr object;
        if (entry["Type"] == "Sphere") {
            auto sphere = std::make_shared<Sphere>();
            sphere->position = read_vec3(entry["Position"]);
            sphere->radius = entry["Radius"];
            object = sphere;
        } else if (entry["Type"] == "Parallelogram") {
            // Parallelogram compatibility
            auto parallelogram = std::make_shared<Parallelogram>();
            parallelogram->origin = read_vec3(entry["Origin"]);
            parallelogram->u = read_vec3(entry["U"]);
            parallelogram->v = read_vec3(entry["V"]);
            object = parallelogram;
        } else if (entry["Type"] == "Mesh") {
            // Load mesh from a file
            std::string filename = std::string(DATA_DIR) + entry["Path"].get<std::string>();
            object = std::make_shared<Mesh>(filename);
        }
        object->material = scene.materials[entry["Material"]];
        scene.objects.push_back(object);
    }

    return scene;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " scene.json" << std::endl;
        return 1;
    }
    Scene scene = load_scene(argv[1]);
    render_scene(scene);
    return 0;
}
