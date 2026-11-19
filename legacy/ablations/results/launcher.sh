cd /home/user/Aether/legacy
for slot in 1 2 3 4; do
  while [ $(pgrep -fc "python3 -W ignore ablations/") -ge 4 ]; do sleep 15; done
  nohup ablations/jobq.sh > /dev/null 2>&1 &
  sleep 20
done
wait
