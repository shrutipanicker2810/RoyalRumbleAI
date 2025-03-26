# RoyalRumbleAI - Wresting Simulation
We are creating an autonomous wrestling simulation, inspired by WWE's Royal Rumble.

## Instructions to set-up environment (conda is preferred) and install required libraries

python version = 3.9.21

## Python Libraries Required: 
- mujoco==3.1.6
- pandas
- numpy
- gym
- glfw

## Install Required Linux Packages (OpenGL Support) for Simulation
'''
sudo apt update
sudo apt install -y libgl1-mesa-glx libgl1-mesa-dri libglapi-mesa mesa-utils
sudo add-apt-repository universe
sudo apt install -y mesa-utils mesa-utils-bin
'''

### Test OpenGL
glxinfo | grep "OpenGL version"
which glxgears

## Set Up Display Server on Windows (VcXsrv)
Download and install VcXsrv:
https://sourceforge.net/projects/vcxsrv/

Launch XLaunch with the following settings:
- Select "Multiple windows"
- Display Number: 0
- Check "Disable access control"
- Finish and save the configuration

## Configure DISPLAY Variable in WSL

export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

echo 'export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk "{print \$2}"):0' >> ~/.bashrc
source ~/.bashrc

echo $DISPLAY

If the 'DISPLAY' variable does not return a MAC-address, then we can replace it with Windows IP 
- ipconfig (in Powershell)

export DISPLAY= Windows IP  # replace Windows IP here

sed -i '/export DISPLAY=/d' ~/.bashrc
echo 'export DISPLAY= Windows IP' >> ~/.bashrc # replace Windows IP here
source ~/.bashrc

## Test X Server with GUI Tools
xclock # a simple GUI clock

## Run this script inline - it will return "GLFW test passed!"

python -c "
import glfw
if not glfw.init():
raise RuntimeError('Failed to initialize GLFW')
window = glfw.create_window(800, 600, 'Test', None, None)
if not window:
glfw.terminate()
raise RuntimeError('Failed to create window')
glfw.make_context_current(window)
glfw.swap_buffers(window)
glfw.poll_events()
glfw.terminate()
print('GLFW test passed!')
"

## Run "run_match.py" for a match between 2 players only and "run_battle_royale.py" for a complete match season with all players.

