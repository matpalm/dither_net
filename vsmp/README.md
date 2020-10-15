cd ~/vsmp/
find frames/ -type f -name \*png | sort > manifest.txt
# edit vsmp.service to change config as required (e.g. --time-between-frames)
sudo cp vsmp.service /lib/systemd/system/
sudo systemctl enable vsmp.service
sudo systemctl start vsmp.service