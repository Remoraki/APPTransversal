using UnityEngine;

public class PerlinNoise : MonoBehaviour
{

    public int width = 256;
    public int height = 256;

    public float scale = 20.0f;

    public float offsetX = 100.0f;
    public float offsetY = 100.0f;

    private Texture2D noiseTexture;
    private Color[] pixels;
    private Renderer rend;

    void Start()
    {
        rend = GetComponent<Renderer>();
        noiseTexture = new Texture2D(width, height);
        pixels = new Color[width * height];
        rend.material.mainTexture = noiseTexture;   
    }

    void CalcNoise()
    {
        for (float y = 0.0f; y < noiseTexture.height; y++)
        {
            for (float x = 0.0f; x < noiseTexture.width; x++)
            {
                float xCoord = x / noiseTexture.width * scale + offsetX;
                float yCoord = y / noiseTexture.height * scale + offsetY;
                float sample = Mathf.PerlinNoise(xCoord, yCoord);
                pixels[(int)y * noiseTexture.width + (int)x] = new Color(sample, sample, sample);
            }
        }

        noiseTexture.SetPixels(pixels);
        noiseTexture.Apply();
    }

    void Update()
    {
        CalcNoise();
    }
}
