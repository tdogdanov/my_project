from my_cv import read_image, show_image, convert_to_gray, play_video, play_camera

# Работа с изображением
img = read_image('example.jpg')
gray = convert_to_gray(img)
show_image(gray, 'Gray Image')

# Работа с видео
play_video('sample.mp4')

# Работа с камерой
play_camera(0)
