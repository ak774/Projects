import pyttsx3
import datetime
import speech_recognition as sr
import pyaudio
import wikipedia
import webbrowser
import os
import smtplib
engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
#print(voices[0].id)
engine.setProperty('voice',voices[0].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
def wishme():
    hour=int(datetime.datetime.now().hour)
    if (hour>=0 and hour<12):
        speak("Good Morning!")
    elif (hour>=12 and hour<18):
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I Am Jarvis,Sir Please Tell Me How May i help you.")
def takecommand():
    r=sr.Recognizer()
    with sr.Microphone(device_index=1) as source:
        print("Listening...")
        r.pause_threshold=1
        r.adjust_for_ambient_noise(source)
        audio=r.listen(source)
    try:
        print("recognizing")
        query=r.recognize_google(audio,language='en-IN')
        print(f'user said {query}\n')
    except Exception as e:
        print("Say That Again Please...")
        return "None"
    return query
def sendEmail(to,content):
    server=smtplib.SMTP('smtp.gmail.com',587)
    server.ehlo()
    server.starttls()
    server.login('<your id>','<password>')
    server.sendmail('<reciever id>',to,content)
    server.close()



if __name__=='__main__':
    wishme()
    while True:
        query=takecommand().lower()

        #searching wikipedia
        if 'wikipedia' in query:
            speak("Searching Wikipedia...")
            query=query.replace("wikipedia","")
            results=wikipedia.summary(query,sentences=2)
            speak("According to Wikipedia")
            print(results)
            speak(results)
        elif 'open youtube' in query:

            url = 'youtube.com'
            webbrowser.register('chrome',
                                None,
                                webbrowser.BackgroundBrowser(
                                    "C://Program Files (x86)//Google//Chrome//Application//chrome.exe"))
            webbrowser.get('chrome').open(url)

        elif 'open google' in query:

            url = 'google.com'
            webbrowser.get('chrome').open(url)
        elif 'open github' in query:

            url = 'github.com'
            webbrowser.get('chrome').open(url)
        elif 'play movie' in query:
            movie_d='C:\\Users\\akshay goel\\Videos\\Captures'
            movies=os.listdir(movie_d)
            os.startfile(os.path.join(movie_d,movies[0]))



        elif 'the time' in query:
            strttime=datetime.datetime.now().strftime("%H:%M:%S")
            speak(f'Sir, the time is {strttime}')
        elif 'open code' in query:
            codepath='C:\\Users\\akshay goel\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe'
            os.startfile(codepath)
        elif 'send email to akshay' in query:
            try:
                speak('what should i say')
                content=takecommand()
                to="goelakshay222@gmail.com"
                sendEmail(to,content)
                speak("Email has been sent")
            except Exception as e:
                print(e)
                speak("Sorry sir, email could not be sent")
        elif 'play music' in query:
            os.system("spotify")
        elif 'thanks jarvis' in query:
            speak('My pleasure sir')
            exit()