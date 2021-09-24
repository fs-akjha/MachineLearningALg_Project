import speech_recognition as sr

r=sr.Recognizer()
print("Please Talk....")
with sr.Microphone() as source:
    audio_data=r.listen(source,phrase_time_limit=10)
    print("Recognizing Microphone input") 
    text=r.recognize_google(audio_data)
    print("Recognized Text is....")
    print(text)