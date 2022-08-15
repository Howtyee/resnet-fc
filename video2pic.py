import cv2

video_path = 'C:/Users/42516/Desktop/数据/1.mp4'  # 视频地址
output_path = 'C:/Users/42516/Desktop/pic/'  # 输出文件夹


if __name__ == '__main__':
    num = 1
    vid = cv2.VideoCapture(video_path)
    while vid.isOpened():
        is_read, frame = vid.read()
        if is_read:

            file_name = '%08d' % num
            cv2.imwrite(output_path +'d_'+ str(file_name) + '.jpg', frame)
            # 00000111.jpg 代表第111帧
            cv2.waitKey(1)
            num += 1

        else:
            break
