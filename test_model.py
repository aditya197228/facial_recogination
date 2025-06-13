from keras.models import load_model

print("🚀 Starting model load...")

try:
    model = load_model("model/facenet_keras.h5")
    print("✅ Model loaded successfully!")
    model.summary()
except Exception as e:
    print("❌ Error loading model:", e)
