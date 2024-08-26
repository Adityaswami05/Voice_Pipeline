import ssl
import whisper
import certifi
import torch
import librosa
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import edge_tts
import asyncio
import nest_asyncio
import warnings

# Apply the workaround to handle the event loop issue in Jupyter/Colab
nest_asyncio.apply()

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set up SSL context to handle SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Step 1: Voice Query to Text (Transcription)
def load_and_preprocess_audio(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        # Load the audio file using librosa, resampling to 16 kHz
        audio, sample_rate = librosa.load(file_path, sr=16000)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")

    # Convert the audio to a tensor and add a batch dimension
    audio = torch.tensor(audio).unsqueeze(0)
    return audio

def transcribe_audio(audio):
    # Load the Whisper model
    model = whisper.load_model("base.en")

    # Transcribe the audio
    audio_numpy = audio.squeeze().numpy()
    result = model.transcribe(audio_numpy)
    return result['text']

# Step 2: Generate a Response Using LLM
def load_model_and_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        raise

def generate_response(tokenizer, model, query):
    try:
        # Tokenize the input query
        inputs = tokenizer.encode(query, return_tensors='pt')

        # Generate the response using the model
        outputs = model.generate(
            inputs,
            max_length=100,                # Ensure the response is detailed
            min_length=100,                # Set a minimum length of 100 words
            no_repeat_ngram_size=2,        # Prevent repetition
            num_return_sequences=1,        # Generate a single sequence
            temperature=0.7,               # Controls creativity; lower values are less creative
            top_k=50,                      # Limits the sampling pool to top k tokens
            early_stopping=True,           # Stop early when EOS token is reached
            pad_token_id=tokenizer.eos_token_id,  # Ensure padding uses the EOS token
            attention_mask=torch.ones(inputs.shape, dtype=torch.long)  # Explicitly set attention mask
        )

        # Decode the generated tokens into text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Step 3: Convert Text to Speech
async def text_to_speech(text, output_file, voice="en-US-GuyNeural"):
    try:
        # Restrict the output to a maximum of 2 sentences
        sentences = text.split('.')
        text = '. '.join(sentences[:2]) + '.'

        # Initialize the Communicate object with optional voice settings
        communicate = edge_tts.Communicate(
            text,
            voice=voice
        )
        await communicate.save(output_file)
        print(f"Speech saved to {output_file}")
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")

if __name__ == "__main__":
    # Define the audio file path
    audio_file = '/Users/adityaswami/Desktop/End-to-End AI Voice Assistance Pipeline/Melakottaiyur 2.wav'  # Replace with your audio file path

    # Step 1: Transcribe audio to text
    transcribed_text = None
    try:
        audio = load_and_preprocess_audio(audio_file)
        transcribed_text = transcribe_audio(audio)
        print(f"Transcribed Text: {transcribed_text}")
    except Exception as e:
        print(f"Error in transcription: {e}")

    # Step 2: Generate a response from the LLM
    generated_response = None
    if transcribed_text:
        try:
            model_name = "gpt2"
            tokenizer, model = load_model_and_tokenizer(model_name)
            generated_response = generate_response(tokenizer, model, transcribed_text)
            print(f"Generated Response: {generated_response}")
        except Exception as e:
            print(f"Error in generating response: {e}")

    # Step 3: Convert the generated response to speech
    if generated_response:
        try:
            output_file = "output.mp3"  # Define the output file path
            asyncio.run(text_to_speech(
                text=generated_response,
                output_file=output_file,
                voice="en-US-GuyNeural"  # Male voice
            ))
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")