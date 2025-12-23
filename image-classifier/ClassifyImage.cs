using System.Net;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using ImageSharpImage = SixLabors.ImageSharp.Image;

namespace image_classifier;

public class ClassifyImage {
    private static readonly string ModelPath =
        Path.Combine(AppContext.BaseDirectory, "model", "efficientnet-lite4-11.onnx");
    private static readonly string LabelsPath =
        Path.Combine(AppContext.BaseDirectory, "model", "labels.txt");

    private static readonly Lazy<InferenceSession> Session = new(() => new InferenceSession(ModelPath));
    private static readonly Lazy<string[]> Labels = new(() => File.ReadAllLines(LabelsPath));

    [Function("ClassifyImage")]
    public async Task<HttpResponseData> Run(
        [HttpTrigger(AuthorizationLevel.Anonymous, "post")] HttpRequestData req) {
        var bytes = await ReadBody(req);
        if (bytes.Length == 0) {
            var bad = req.CreateResponse(HttpStatusCode.BadRequest);
            await bad.WriteAsJsonAsync(new { error = "Send raw image bytes in the POST body (binary)." });
            return bad;
        }

        using Image<Rgb24> img = ImageSharpImage.Load<Rgb24>(bytes);
        img.Mutate(x => x.Resize(224, 224));

        // EfficientNet-Lite4 ONNX expects NHWC float32: [1,224,224,3]
        var inputName = Session.Value.InputMetadata.Keys.First();
        var outputName = Session.Value.OutputMetadata.Keys.First();

        var input = new DenseTensor<float>(new[] { 1, 224, 224, 3 });
        FillNhwcRgbMinus127Div128(img, input);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, input)
        };

        using var results = Session.Value.Run(inputs);
        var logitsOrProbs = results.First(r => r.Name == outputName).AsEnumerable<float>().ToArray();

        // Many EfficientNet-Lite exports already output softmax; keep it safe:
        var probs = SoftmaxIfNeeded(logitsOrProbs);
        int idx = ArgMax(probs);

        var ok = req.CreateResponse(HttpStatusCode.OK);
        await ok.WriteAsJsonAsync(new {
            label = idx < Labels.Value.Length ? Labels.Value[idx] : $"class_{idx}",
            confidence = probs[idx],
            index = idx
        });
        return ok;
    }

    private static async Task<byte[]> ReadBody(HttpRequestData req) {
        using var ms = new MemoryStream();
        await req.Body.CopyToAsync(ms);
        return ms.ToArray();
    }

    // Per EfficientNet-Lite4 README: convert jpg [0..255] -> float [-1..1] via (x-127)/128. :contentReference[oaicite:1]{index=1}
    private static void FillNhwcRgbMinus127Div128(Image<Rgb24> img, DenseTensor<float> t) {
        img.ProcessPixelRows(accessor => {
            for (int y = 0; y < 224; y++) {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < 224; x++) {
                    var p = row[x];
                    t[0, y, x, 0] = (p.R - 127f) / 128f;
                    t[0, y, x, 1] = (p.G - 127f) / 128f;
                    t[0, y, x, 2] = (p.B - 127f) / 128f;
                }
            }
        });
    }

    private static int ArgMax(float[] v) {
        int idx = 0;
        float best = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > best) { best = v[i]; idx = i; }
        return idx;
    }

    private static float[] SoftmaxIfNeeded(float[] v) {
        // If it already looks like probabilities (sum ~1 and in [0,1]), return as-is
        double sum = 0;
        bool inRange = true;
        for (int i = 0; i < v.Length; i++) {
            sum += v[i];
            if (v[i] < -0.001 || v[i] > 1.001) inRange = false;
        }
        if (inRange && Math.Abs(sum - 1.0) < 0.05) return v;

        // else softmax
        float max = v.Max();
        var exps = new float[v.Length];
        double s = 0;
        for (int i = 0; i < v.Length; i++) {
            double e = Math.Exp(v[i] - max);
            exps[i] = (float)e;
            s += e;
        }
        for (int i = 0; i < exps.Length; i++)
            exps[i] = (float)(exps[i] / s);
        return exps;
    }
}
