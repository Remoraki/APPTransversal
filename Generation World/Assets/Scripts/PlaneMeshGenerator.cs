using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class PlaneMeshGenerator : MonoBehaviour {
    public int width = 100;
    public int height = 100;
    public float scale = 1f;
    public GameObject perlinNoiseObject;

    void Start() {
        GenerateMesh();
    }

    void GenerateMesh() {
        if (perlinNoiseObject == null) {
            Debug.LogError("PerlinNoise object is not set.");
            return;
        }

        PerlinNoise perlinNoise = perlinNoiseObject.GetComponent<PerlinNoise>();
        if (perlinNoise == null) {
            Debug.LogError("PerlinNoise component is not found.");
            return;
        }

        Texture2D heightMap = perlinNoise.GetTexture();

        MeshFilter meshFilter = GetComponent<MeshFilter>();
        Mesh mesh = new Mesh();

        Vector3[] vertices = new Vector3[(width + 1) * (height + 1)];
        int[] triangles = new int[width * height * 6];
        Vector2[] uv = new Vector2[vertices.Length];

        int t = 0;
        for (int y = 0; y <= height; y++) {
            for (int x = 0; x <= width; x++) {
                int index = y * (width + 1) + x;
                float heightValue = 0f;
                if (heightMap)
                    heightValue = heightMap.GetPixelBilinear((float)x / width, (float)y / height).grayscale;
                
                vertices[index] = new Vector3(x * scale, heightValue * 10, y * scale);
                uv[index] = new Vector2((float)x / width, (float)y / height);

                if (x < width && y < height) {
                    triangles[t++] = index;
                    triangles[t++] = index + width + 1;
                    triangles[t++] = index + 1;

                    triangles[t++] = index + 1;
                    triangles[t++] = index + width + 1;
                    triangles[t++] = index + width + 2;
                }
            }
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.uv = uv;
        mesh.RecalculateNormals();

        meshFilter.mesh = mesh;
    }

    void Update() {
        
    }
}
