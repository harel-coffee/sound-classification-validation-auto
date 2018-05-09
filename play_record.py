from naoqi import ALProxy
import glob
from playsound import playsound
import time
from sklearn.cross_validation import train_test_split
import pickle

files = glob.glob('./Dataset_mono_16k/*.wav')
classes = []
for audio_file in files:
    classes.append((audio_file.split('__')[1]).split('_')[0])

_, to_play, _, classes_played = train_test_split(files, classes, test_size=0.25, random_state=None)

audio_recorder = ALProxy("ALAudioRecorder", "yvette.local", 9559)
audio_recorder.stopMicrophonesRecording()
# print(to_play)
# print(classes_played)

with open('classes_played_nao.sav', 'wb') as f:
    pickle.dump((classes_played, to_play), f)

print('Total files %s' % len(classes_played))
size = len(to_play)
time_start = time.time()

channels = [0, 0, 1, 0]
# time.sleep(2)
# audio_recorder.startMicrophonesRecording('/home/nao/naoqi/recordings/nao_silence.wav', "wav", 16000, channels)
# time.sleep(5)
# audio_recorder.stopMicrophonesRecording()

for i in range(len(to_play)):
    filename = '/home/nao/naoqi/recordings/recording_%s.wav' % i
    audio_recorder.startMicrophonesRecording(filename, "wav", 16000, channels)
    time.sleep(0.25)
    playsound(to_play[i])
    time.sleep(0.25)
    audio_recorder.stopMicrophonesRecording()
    print('Completed %.2f%% in %.2f s' % (100.*(i+1)/size, time.time()-time_start))
# playsound('/home/fpetric/devel/code/workspace/python-libxtract/Dataset_mono_16k/1_year_old_babble__babbling_001.wav')
# time.sleep(0.5)
# playsound('/home/fpetric/devel/code/workspace/python-libxtract/Dataset_mono_16k/1_year_old_babble__babbling_001.wav')