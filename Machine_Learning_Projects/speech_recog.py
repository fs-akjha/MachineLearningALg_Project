import speech_recognition as sr

filename='Perfect - Ed Sheeran- [MyMp3Bhojpuri.In].wav'
r=sr.Recognizer()
with sr.AudioFile(filename) as source:
    audio_data=r.listen(source)
    try:
        text=r.recognize_google(audio_data)
        print("Recognizing...")
        print(text)
    except:
        print("sorry, run again...")