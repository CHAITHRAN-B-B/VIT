import { GoogleGenAI, createUserContent, createPartFromUri } from "@google/genai";

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;
    const text = formData.get("text") as string | null;

    const apiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY;
    if(!apiKey){
      throw new Error("API Key is not defined")
    }
    const ai = new GoogleGenAI({apiKey});

    // 🟢 CASE 1 — Normal text conversation
    if (!file) {
      if (!text) {
        return new Response("No input provided", { status: 400 });
      }

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: text,
      });

      return Response.json({ text: response.text });
    }

    // 🟢 CASE 2 — Image uploaded

    // Convert File → Buffer
    const buffer = Buffer.from(await file.arrayBuffer());

    // 1️⃣ Upload image to Gemini File API
    const uploaded = await ai.files.upload({
      file: file,
      config: { mimeType: file.type },
    });

    // 2️⃣ Call FastAPI ViT
    const vitForm = new FormData();
    vitForm.append("file", file);

    const vitRes = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: vitForm,
    });

    const vitData = await vitRes.json();

    // 3️⃣ Ask Gemini to describe + justify classification
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: createUserContent([
        createPartFromUri(uploaded.uri!, uploaded.mimeType!),
        `The Vision Transformer classified this image as ${vitData.prediction}
with ${vitData.confidence}% confidence.

Use this as ground truth.

1. First describe what you see in the image.
2. Clearly state the classification.
3. Explain why it may be real or AI-generated.

Keep under 150 words.`,
      ]),
    });

    return Response.json({ text: response.text });

  } catch (error: any) {
    console.error("Gemini Error:", error);
    return new Response("Internal Server Error", { status: 500 });
  }
}
