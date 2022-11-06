Хочу поделиться опытом придания Yandex Алисе (внутри умных колонок поддерживающих TTS API) своевольности. Про локальный API колонок уже писали на хабре тут https://habr.com/ru/post/508106/

Идея заключается в том чтобы Алиса реагировала на присутствующих людей. Для этого их необходимо идентифицировать. Так же Алиса в случайное время будет выдавать некоторую «отсебятину». В репозитории использован самый простой (на мой взгляд) вариант – тренировка модели в Google Teachable Machine. Использован модуль для Node-Red который позволяет посылать команды, управлять воспроизведением и отправлять на яндекс станцию текст который она произнесет.

Инструменты:
1.	Docker
2.	Node-Red
3.	Python3
4.	OpenCV
5.	Tensorflow Keras
6.	Google Teachable Machine
7.	PostgresSQL
8.	MQTT


![Sheme](/images/sheme.png)

1)	Скрипт распознает лица и отправляет процент совпадения для каждой сущности модели в MQTT брокер, а так же сохраняет снимок в директорию из файла конфигурации

2)	Node-Red получает сообщения из MQTT брокера и отправляет Алисе команду что либо сказать в зависимости от времени суток и того как давно она “видела” этого человека. При этом фразы для Алисы, а так же время когда человек последний раз засветился на камеру хранятся в БД Postgres.

# Установка

**Linux:**
```
pip3 install opencv-python tensorflow
```
Возможно портебуется установка дополнительных пакетов
```
sudo apt-get install git python3-dev 
```
Загружаем приложение
```
git clone https://github.com/guinmoon/yandex_alice_has_fun
cd yandex_alice_has_fun 
```
```keras_model.h5``` и ```labels.txt``` меняем на свои


# Запуск
В файле .env меняем путь на расположение yandex_alice_has_fun
```
docker-compose up
```
Если Postgers, Mosquitto и Node-Red запустились без ошибок то
```
docker-compose up -d
python3 alice_has_fun.py
```
При первом запуске в Node-Red необходимо установить дополнительные модули ```node-red-contrib-postgresql, node-red-contrib-yandex-station-management, node-red-dashboard```. Сделать это лучше всего через Настройки -> Управление палитрой. Должно получиться так:

![modules](/images/red_modules.png)

После этого необходимо ипортировать файл ```data/alice_has_fun_flows.json``` в Node-Red и настроить соединение с колонкой и БД
# Настройка
Настройки alice_has_fun.py хранятся в файле config.json:
```
{
    "debug": false,
    "show_camera": false,
    "input_sounrce": 0,
    "frame_recognition_rate": 4,
    "publish_min_dalay": 1,
    "camera_w": 1080,
    "camera_h": 720,
    "mqtt_broker": "127.0.0.1",
    "save_on_detect": true
}
```
| Параметр               |                                                              Назначение                                                              |
| ---------------------- | :----------------------------------------------------------------------------------------------------------------------------------: |
| debug                  |                                         Елси True то рзультаты не публикуются в MQTT брокер                                          |
| show_camera            |                                  Елси True то с помощью cv2.imshow отображается видео из источника                                   |
| input_sounrce          | Число (0 это первая веб камера), адрес потока ("rtsp://192.168.1.86:8554/unicast") или расположение видео файла ("raw/IMG_2404.MOV") |
| frame_recognition_rate |                                                Частота распознавания кадров в секунду                                                |
| publish_min_dalay      |                            Задержка перед публикацией, публикуются средние значения из распознаных кадров                            |
| camera_w               |                                  Ширина видео для устройств поддерживающих задание разрешения видео                                  |
| camera_h               |                                  Высота видео для устройств поддерживающих задание разрешения видео                                  |
| mqtt_broker            |                                                          Адрес MQTT брокера                                                          |
| save_on_detect         |                             Помимо публикации в брокер также будет сохранен кадр в директорию on_detect                              |

# Автозапуск
Для корректного запуска docker-copmpose при старте системы лучше создать сервис, чтобы убедиться что необходимые службы запущены
```
sudo nano /etc/systemd/system/alice-docker-compose.service
```
```
[Unit]
Description=Docker Compose Application Service
Requires=docker.service
After=docker.service

[Service]
WorkingDirectory=<Путь к рабочей директории alice has fun>
ExecStart=/usr/local/bin/docker-compose up
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
Restart=on-failure
StartLimitIntervalSec=60
StartLimitBurst=3

[Install]
WantedBy=multi-user.target
```
```
sudo systemctl enable alice-docker-compose
```
Скрипт alice_has_fun.py можно запускать отдельно, например через crontab, скриптом вроде этого
```
#!/bin/bash
cd /home/m_vs_m/soft/alice_live
if screen -list | grep -q "alice_live"; then
    /usr/bin/screen -S alice_live -X quit
fi
/usr/bin/screen -S alice_live -d -m python3 alice_has_fun.py
```
Отдельный запуск позволяет на время остановить распознавание, снизив нагрузку на железо, но оставляя возможность дорабатывать систему.
Для отсановки необходимо завершить screen сессию
```
if screen -list | grep -q "alice_live"; then
    /usr/bin/screen -S alice_live -X quit
fi
```