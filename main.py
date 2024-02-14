from googleapiclient.discovery import build
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock, mainthread
from PIL import Image as PilImage, ImageGrab
from predict.predict_emotion import predict_expression
import time
from kivy.graphics.texture import Texture
import threading
from plyer import notification
import webbrowser

checkpoint_path = 'C:/Users/dulsh/Documents/PROJECT_EUSL/pythonProject/model/best_model.h5'
model = None 

label_to_text = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

def suggest_songs(emotion):
    api_key = 'AIzaSyCZRf0Dy0VLzumZFJl7kDcMxqyeR7AcwlI'
    youtube = build('youtube', 'v3', developerKey=api_key)

    query = f"{emotion} music"

    request = youtube.search().list(
        q=query,
        part='id,snippet',
        type='video',
        maxResults=5
    )
    response = request.execute()

    videos = []
    for item in response['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        videos.append({'title': video_title, 'id': video_id})

    return videos


class RoundButton(Button):
    def __init__(self, **kwargs):
        super(RoundButton, self).__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)
        self.bind(size=self.update_canvas, pos=self.update_canvas)
        self.canvas.before.add(Color(1, 1, 1, 1))
        self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[10])
        self.width = 100
        self.height = 50

    def update_canvas(self, instance, value):
        self.canvas.before.clear()
        self.canvas.before.add(Color(0.5, 0.5, 0.5, 0.5))
        self.rect.pos = self.pos
        self.rect.size = self.size


class CustomMessagePopup(Popup):
    def __init__(self, video_title, video_id, **kwargs):
        super(CustomMessagePopup, self).__init__(**kwargs)
        self.title = "Song Suggestion"
        self.size_hint = (None, None)
        self.size = (400, 200)

        content = BoxLayout(orientation='vertical', spacing=10, padding=10)

        message_label = Label(text=f"Try this song:\n{video_title}", halign='center', valign='center')
        content.add_widget(message_label)

        youtube_button = Button(text="Open on YouTube", size_hint=(None, None))
        youtube_button.bind(on_press=lambda x: self.open_youtube(video_id))
        content.add_widget(youtube_button)

        self.content = content

    def open_youtube(self, video_id):
        url = f"https://www.youtube.com/watch?v={video_id}"
        webbrowser.open(url)


class MusicalApp(App):
    def build(self):
        self.icon_text_layout = BoxLayout(orientation='vertical')

        self.icon_image = Image(source='C:/Users/dulsh/Documents/PROJECT_EUSL/pythonProject/image/iconimage.png',
                                size=(200, 200),
                                pos_hint={'center_x': 0.5})
        self.icon_text_layout.add_widget(self.icon_image)

        text_label = Label(text="TUNE MOOD", halign='center', valign='center', font_size='14sp', color=[1, 1, 1, 1])
        self.icon_text_layout.add_widget(text_label)

        self.toggle_button = RoundButton(text='START', font_size='20sp', on_press=self.toggle_button_press,
                                         background_color=[0.18, 0.18, 0.18, 1], background_normal='')
        self.icon_text_layout.add_widget(self.toggle_button)

        self.custom_message_popup = CustomMessagePopup("", "")

        self.thread_running = False

        return self.icon_text_layout

    def toggle_button_press(self, instance):
        if instance.text == 'START':
            instance.text = 'STOP'
            self.thread_running = True
            threading.Thread(target=self.run_capture_thread).start()
        else:
            instance.text = 'START'
            self.thread_running = False

    def run_capture_thread(self):
        while self.thread_running:
            self.capture_photo(None)
            time.sleep(5)

    @mainthread
    def update_custom_message_popup(self, suggested_song):
        self.custom_message_popup.title = suggested_song['title']
        self.custom_message_popup.content.children[1].bind(
            on_press=lambda x: self.open_youtube(suggested_song['id']))
        self.custom_message_popup.open(pos_hint={'center_x': 0.5, 'center_y': 0.5})
        
        notification_title = 'TuneMood Applet'
        notification_message = f"Hey! This is TuneMood Applet. Try this song:\n\n{ suggested_song['title'] }"
        notification.notify(
            title=notification_title,
            message=notification_message,
            timeout=5
        )

    def capture_photo(self, dt):
        screenshot = ImageGrab.grab()
        screenshot_bytes = screenshot.tobytes()
        texture = Texture.create(size=(screenshot.width, screenshot.height), colorfmt='rgb')
        texture.blit_buffer(screenshot_bytes, colorfmt='rgb', bufferfmt='ubyte')
        
        self.icon_image.texture = texture

        pil_image = PilImage.frombytes("RGB", screenshot.size, screenshot_bytes)
        emotion = predict_expression(pil_image)
        print(f"Predicted emotion: {emotion}")

        suggested_songs = suggest_songs(emotion)
        print("Suggested Songs:")
        for song in suggested_songs:
            print(f"Title: {song['title']}")
            print(f"Video ID: {song['id']}")
            print("------")

        self.update_custom_message_popup(suggested_songs[0])

        self.thread_running = False

    def open_youtube(self, video_id):
        url = f"https://www.youtube.com/watch?v={video_id}"
        webbrowser.open(url)


if __name__ == '__main__':
    MusicalApp().run()
