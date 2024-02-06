''' 
Pouring over side data collection 
Alex Gillespie
08/03/2023
 '''

from xarm.wrapper import XArmAPI
import os
import serial
import numpy as np
import pickle
import time
import threading
import queue
import torch
from model import Transformer, MLP, CNN1D, MLP_2, MLP_2_class, MLP_2_10_resnet, MLP_2_10_resnet_class
from scipy.optimize import fsolve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

arm = XArmAPI("192.168.1.199")
# TODO
# SPECIFY: the directory you're working in
current_dir = r'C:\Users\yexin\Desktop\liquid\data'

weight_data_queue = queue.Queue()
# sets up sensors


np.random.seed(0)
torch.manual_seed(412)
model = MLP_2_10_resnet_class()
device = 'cuda'
model.to(device)
checkpoint = torch.load(r"C:\Users\yexin\Desktop\liquid\ckpts_final\150_2.path.tar")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def linear_interpolation(full_time, time, weight):
    interpolated_weights = np.interp(full_time, time, weight)
    return interpolated_weights


def collect_weight_data(serial_weight, stop_event):
    while not stop_event.is_set():
        line = serial_weight.readline()
        try:
            decoded_line = line.decode('ascii').split(',')[0]
            if decoded_line.split(":")[0] == "grams":
                float_value = float(decoded_line.split(":")[1].replace("\r\n", ""))
                weight_data_queue.put(float_value)
        except UnicodeDecodeError:
            pass


def findFrame(ts_target, ts_set):
    idx = (np.abs(ts_target-ts_set)).argmin()
    return idx


def setup_sensors():
    # Serial initialization
    baudrate = 115200
    # TODO
    # SPECIFY: your serial ports for the Teensys
    port1 = 'COM5'
    port2 = 'COM3'
    port_weight = 'COM4'
    buffer1 =  [] # for values of teensy 1
    buffer2 =  [] # for values of teensy 2
    buffer_weight = [] # for values of weight from Arduino uno
    serialTeensy_1 = serial.Serial(port1, baudrate)
    serialTeensy_2 = serial.Serial(port2, baudrate)
    serial_weight = serial.Serial(port_weight, 57600) #57600
    # checking on Teensy connections and resetting input buffer
    if serialTeensy_1 is None:
        raise RuntimeError('Serial Port is not found!')
    serialTeensy_1.reset_input_buffer()
    if serialTeensy_2 is None:
        raise RuntimeError('Serial Port is not found!')
    serialTeensy_2.reset_input_buffer()
    if serial_weight is None:
        raise RuntimeError('Serial Port is not found!')
    serial_weight.reset_input_buffer()
    return buffer1, buffer2, buffer_weight, serialTeensy_1, serialTeensy_2, serial_weight

# data collection process for pouring
def collect_data_pour(container, content, labels, iteration, rotate_speed, rotate_angle, p_duration, grip_height, grasp_height, p1_duration, final_weight):

    
    buffer1, buffer2, buffer_weight, serialTeensy_1, serialTeensy_2, serial_weight = setup_sensors()
    weight_time = []
    weight_pre = []
    weight_pre_time = []
    
    time.sleep(5)
    #buffer1, buffer2, buffer_weight, serialTeensy_1, serialTeensy_2 = setup_sensors()
    count = 0
    flag_first_time = True
    stop_event = threading.Event()
    weight_thread = threading.Thread(target=collect_weight_data, args=(serial_weight, stop_event))
    weight_thread.start()
    
    arm.set_position(410, 16.7, 260, rotate_angle[0], rotate_angle[1], rotate_angle[2], speed=rotate_speed, is_radian=False, wait=False)
    start = time.time()
    count_new = 0
    performed_action = False
    weight = 0
    array_record = 0
    # while the arm is moving, collect data
    while time.time() - start <= p_duration:

            

        buffer1.append(serialTeensy_1.read())
        buffer2.append(serialTeensy_2.read())
        
        try:
            float_value = weight_data_queue.get_nowait()  # non-blocking get
            buffer_weight.append(float_value)
            print(float_value)

            weight_time.append(time.time() - start)
        except queue.Empty:
            pass

        if len(buffer1) >= 18 and len(buffer2) >= 18:
            # These bytes indicate a new series of sensor values 
            new_line1 = int.from_bytes(buffer1[0], byteorder='little')
            new_line2 = int.from_bytes(buffer1[1],byteorder='little')
            new_line3 = int.from_bytes(buffer1[2],byteorder='little')
            new_line4 = int.from_bytes(buffer2[0], byteorder='little')
            new_line5 = int.from_bytes(buffer2[1],byteorder='little')
            new_line6 = int.from_bytes(buffer2[2],byteorder='little')
            # If all the bytes are 255, the maximum value (3 max bytes in a row for each teensy), that's how we know we're reading the sensor values
            if new_line1 == 255 and new_line2 == 255 and new_line3 == 255 and new_line4 == 255 and new_line5 == 255 and new_line6 == 255:
                # These teensy values store the time at which the values were taken 
                teensy_1_time = int.from_bytes(buffer1[13]+buffer1[14]+buffer1[15]+buffer1[16],byteorder='little')
                teensy_2_time = int.from_bytes(buffer2[13]+buffer2[14]+buffer2[15]+buffer2[16],byteorder='little')
                duration = time.time()-start
                # headers: container, content, labels, iteration, observation (in iteration), python time passed, Teensy time for sensor 1, Teensy time for sensor 2, sensors 1-10
                sensor_vals = np.array([container, content, labels, iteration, count, duration, teensy_1_time, teensy_2_time,
                int.from_bytes(buffer1[3]+buffer1[4],byteorder='little'), 
                int.from_bytes(buffer1[5]+buffer1[6],byteorder='little'), 
                int.from_bytes(buffer1[7]+buffer1[8],byteorder='little'), 
                int.from_bytes(buffer1[9]+buffer1[10],byteorder='little'),
                int.from_bytes(buffer1[11]+buffer1[12],byteorder='little'),
                int.from_bytes(buffer2[3]+buffer2[4],byteorder='little'), 
                int.from_bytes(buffer2[5]+buffer2[6],byteorder='little'), 
                int.from_bytes(buffer2[7]+buffer2[8],byteorder='little'), 
                int.from_bytes(buffer2[9]+buffer2[10],byteorder='little'),
                int.from_bytes(buffer2[11]+buffer2[12],byteorder='little')])
                # if this is the first sensor reading of the iteration, it creates a new array
                if flag_first_time:
                    array_record = sensor_vals
                    flag_first_time = False
                # if this is not the first sensor reading of the iteration, it adds on the values to the existing array
                else:
                    array_record = np.vstack((array_record, sensor_vals))
                    
                    # print('Frequency: ', count/(time.time()-start)) # you'll watch these in the terminal to make sure the times all look like they're consistent
                    # print('Python time: ', (time.time()-duration))
                    # print('Teensy duration 1: ', teensy_1_time-prev_teensy_1_time)
                    # print('Teensy duration 2: ', teensy_2_time-prev_teensy_2_time)
                    
                prev_teensy_1_time = teensy_1_time
                prev_teensy_2_time = teensy_2_time
                count +=1
            buffer1.pop(0)
            buffer2.pop(0)
            
        if time.time() - start >= 0.2:
            if (len(array_record) - 500) % 10 == 0 and count > count_new and not performed_action:
                
                array_record_p = array_record[500:]
                s = array_record_p[-10:, 8 : 18].astype(np.float32)
                s = (s - 500) / (1000 - 500)
                cap = torch.tensor(s.reshape((1, 10, 10))).to(device)
                w, offset_1, offset_2 = model(cap, torch.tensor(np.array([4]).astype(np.int64)).to(device))
                
                #w, offset_1, offset_2 = model(cap)
                weight += w.data.item()
                count_new = count
                weight_pre_time.append(time.time() - start + offset_2.data.item() / 100)
                weight_pre.append(weight)
                print(weight)
                

            if (weight >= final_weight or time.time() - start >= p_duration) and not performed_action:
                arm.set_state(4)
                arm.set_state(0)
                performed_action = True
                
                arm.set_position(410, 16.7, 260, 44.9, -87.8, -134.9, speed=50, is_radian=False, wait=False)
                end_time = time.time() - start

        
        
    stop_event.set()
    weight_thread.join()
    arm.set_position(494, 16.7, 260, 44.9, -87.8, -134.9, speed=150, is_radian=False, wait=True)
    arm.set_position(494, 29.1, 40.1+grip_height + grasp_height, 44.9, -87.8, -134.9, speed=150, is_radian=False, wait=True)
    arm.set_gripper_position(840, wait = True)
    arm.set_position(494, 16.7, 215, 44.9, -87.8, -134.9, speed=150, is_radian=False, wait=True)

    array_record = array_record[500:]
    
    idx = findFrame(float(array_record[0][5]), np.array(weight_time))
    idx_t = findFrame(end_time, np.array(array_record[0:, 5]).astype(np.float64))
    
    print(idx_t)
    
    bh = np.zeros(len(array_record))
    bh[idx_t:] = 1
    

    
    weight_array = np.column_stack((weight_time, buffer_weight))
    weight_pre_array = np.column_stack((weight_pre_time, weight_pre))
    weight_array = weight_array[idx:]
    weight_inter = linear_interpolation(array_record[:, 5].astype(np.float32), weight_array[:, 0], weight_array[:, 1])
    weight_pre_inter = linear_interpolation(array_record[:, 5].astype(np.float32), weight_pre_array[:, 0], weight_pre_array[:, 1])
    return array_record, prev_teensy_1_time, prev_teensy_2_time, weight_inter, weight_pre_inter, bh

# collects data for holding portion 
def collect_data_hold(container, content, labels, iteration, hold_time, array_record, prev_teensy_1_time, prev_teensy_2_time):
    #buffer1, buffer2, buffer_weight, serialTeensy_1, serialTeensy_2, serial_weight = setup_sensors()
    buffer1, buffer2, buffer_weight, serialTeensy_1, serialTeensy_2 = setup_sensors()
    count = 0
    start = time.time()
    duration = 0
    while duration < hold_time:
        buffer1.append(serialTeensy_1.read())
        buffer2.append(serialTeensy_2.read())
        #weight_line = serial_weight.readline()
        # try:
        #     decoded_line = weight_line.decode('ascii').split(',')[0]
        #     if decoded_line.split(":")[0] == "grams":
        #         float_value = float(decoded_line.split(":")[1].replace("\r\n", ""))
        #         buffer_weight.append(float_value)
        #         #print(buffer_weight)
        # except UnicodeDecodeError:
        #     pass
        
        # each series of bytes is 17 bytes long, so both the left and right teensy need to be at least 18 bytes long to start reading it
        if len(buffer1) >= 18 and len(buffer2) >= 18:
            # These bytes indicate a new series of sensor values 
            new_line1 = int.from_bytes(buffer1[0], byteorder='little')
            new_line2 = int.from_bytes(buffer1[1],byteorder='little')
            new_line3 = int.from_bytes(buffer1[2],byteorder='little')
            new_line4 = int.from_bytes(buffer2[0], byteorder='little')
            new_line5 = int.from_bytes(buffer2[1],byteorder='little')
            new_line6 = int.from_bytes(buffer2[2],byteorder='little')
            # If all the bytes are 255, the maximum value (3 max bytes in a row for each teensy), that's how we know we're reading the sensor values
            if new_line1 == 255 and new_line2 == 255 and new_line3 == 255 and new_line4 == 255 and new_line5 == 255 and new_line6 == 255:
                # These teensy values store the time at which the values were taken 
                teensy_1_time = int.from_bytes(buffer1[13]+buffer1[14]+buffer1[15]+buffer1[16],byteorder='little')
                teensy_2_time = int.from_bytes(buffer2[13]+buffer2[14]+buffer2[15]+buffer2[16],byteorder='little')
                duration = time.time()-start
                # headers: container, content, labels, iteration, observation (in iteration), python time passed, Teensy time for sensor 1, Teensy time for sensor 2, sensors 1-10
                sensor_vals = np.array([container, content, labels, iteration, count, duration, teensy_1_time, teensy_2_time,
                int.from_bytes(buffer1[3]+buffer1[4],byteorder='little'), 
                int.from_bytes(buffer1[5]+buffer1[6],byteorder='little'), 
                int.from_bytes(buffer1[7]+buffer1[8],byteorder='little'), 
                int.from_bytes(buffer1[9]+buffer1[10],byteorder='little'),
                int.from_bytes(buffer1[11]+buffer1[12],byteorder='little'),
                int.from_bytes(buffer2[3]+buffer2[4],byteorder='little'), 
                int.from_bytes(buffer2[5]+buffer2[6],byteorder='little'), 
                int.from_bytes(buffer2[7]+buffer2[8],byteorder='little'), 
                int.from_bytes(buffer2[9]+buffer2[10],byteorder='little'),
                int.from_bytes(buffer2[11]+buffer2[12],byteorder='little')])
                array_record = np.vstack((array_record, sensor_vals))
                print('Frequency: ', count/(time.time()-start)) # you'll watch these in the terminal to make sure the times all look like they're consistent
                print('Python time: ', (time.time()-duration))
                print('Teensy duration 1: ', teensy_1_time-prev_teensy_1_time)
                print('Teensy duration 2: ', teensy_2_time-prev_teensy_2_time)
                print()
                prev_teensy_1_time = teensy_1_time
                prev_teensy_2_time = teensy_2_time
                count +=1
            buffer1.pop(0)
            buffer2.pop(0)
            # buffer_weight.pop(0)
    return array_record

# positions arm for pick up
def ready_for_pickup(arm):
    arm.set_gripper_position(840, wait = True)
    arm.set_position(494, 16.7, 215, 44.9, -87.8, -134.9, speed=150, is_radian=False, wait=True)
    arm.set_position(494, 29.1, 80.1, 44.9, -87.8, -134.9, speed=150, is_radian=False, wait=True)

# picks up and pours container
def pick_up_pour(arm, closed_pos, grip_height, iteration, container, content, labels, hold_time, rotate_speed, rotate_angle, grasp_height, p_duration, p1_duration, final_weight):
    arm.set_position(494, 29.1, 40.1 + grip_height + grasp_height, 44.9, -87.8, -134.9, speed=50, is_radian=False, wait=True)
    arm.set_gripper_position(closed_pos, wait = True)
    input('press enter to confirm height')
    arm.set_position(494, 16.7, 260, 44.9, -87.8, -134.9, speed=50, is_radian=False, wait=True)
    arm.set_position(410, 16.7, 260, 44.9, -87.8, -134.9, speed=50, is_radian=False, wait=True)
    sensor_vals, prev_teensy_1_time, prev_teensy_2_time, weight_array, weight_pre, bh = collect_data_pour(container, content, labels, iteration, rotate_speed, rotate_angle, p_duration, grip_height, grasp_height, p1_duration, final_weight)
    #array_record = collect_data_hold(container, content, labels, iteration, hold_time, sensor_vals, prev_teensy_1_time, prev_teensy_2_time)
    return sensor_vals, weight_array, weight_pre, bh

# places container back
def place_back(arm, grip_height, rotate_speed, grasp_height):
    arm.set_position(410, 16.7, 260, 44.9, -87.8, -134.9, speed=rotate_speed, is_radian=False, wait=False)
    arm.set_position(494, 16.7, 260, 44.9, -87.8, -134.9, speed=150, is_radian=False, wait=False)
    arm.set_position(494, 29.1, 40.1+grip_height + grasp_height, 44.9, -87.8, -134.9, speed=150, is_radian=False, wait=False)
    arm.set_gripper_position(840, wait = False)
    arm.set_position(494, 16.7, 215, 44.9, -87.8, -134.9, speed=150, is_radian=False, wait=False)

# returns arm to resting position
def resting_pos(arm):
    arm.set_gripper_position(840, wait = True)
    arm.set_position(338.0, 16.7, 260, 44.9, -87.8, -134.9, speed=150, is_radian=False, wait=True)

# sets height and position for whatever container is being used
def set_height_and_pos(container):
    if container == 'ceramic':
        closed_pos = 660
        grip_height = 28
    elif container == 'plastic':
        closed_pos = 615
        grip_height = 25
    elif container == 'paper':
        closed_pos = 610
        grip_height = 18.2
    elif container == 'foam':
        closed_pos = 625
        grip_height = 0
    elif container == 'silicon':
        closed_pos = 650
        grip_height = 54.8
    elif container == 'glass':
        closed_pos = 730
        grip_height = 26.5
    elif container == 'wood':
        closed_pos = 630
        grip_height = 30.1
    return closed_pos, grip_height

# creates the directories
def create_dirs(current_dir):
    # All the data will go into a specified directory
    direct_label = input('what is the title for the final directory? Sugested: date_purpose, ex: MM.DD_test ')
    new_dir = 'data_collection_'+direct_label
    path = os.path.join(current_dir, new_dir)
    # TODO
    # OPTIONAL SPECIFY: If you're adding the files to a directory that already exists, comment this out
    os.mkdir(path)
    os.chdir(current_dir+'/'+new_dir)

# runs all of the code

def get_parameters(rotate_speed_list, rotate_angle_list, grasp_height_list):
    while True:        
        rotate_command = input("select a rotate speed level, please type 1, 2, 3 to specify the level ")
        if rotate_command == "1":
            rotate_speed = rotate_speed_list[0]
            break
        elif rotate_command == "2":
            rotate_speed = rotate_speed_list[1]
            break
        elif rotate_command == "3":
            rotate_speed = rotate_speed_list[2]
            break
        else:
            print("Wrong level, please select again")
    
    while True:
        rotate_angle_command = input("select a rotate angle level, please type high or low to sepcify the level ")
        if rotate_angle_command == "high":
            rotate_angle = rotate_angle_list[0]
            break
        elif rotate_angle_command == "low":
            rotate_angle = rotate_angle_list[1]
            break
        else:
            print("Wrong level, please select again")
    
    while True:
        grasp_height_command = input("select a grasp height level, please type high or low to sepcify the level ")
        if grasp_height_command == "high":
            grasp_height = grasp_height_list[0]
            break
        elif grasp_height_command == "low":
            grasp_height = grasp_height_list[1]
            break
        else:
            print("Wrong level, please select again")
            
    while True:
        fill_level = input("select a fill level of your current cup, please type high or low to specify the level ")
        if fill_level == "high":
            break
        elif fill_level == "low":
            break
        else:
            print("Wrong level, please select again")
            
    return rotate_speed, rotate_angle, grasp_height, rotate_command, rotate_angle_command, grasp_height_command, fill_level


def trial():
    # TODO 
    # SPECIFY: containers and contents 
    with open(r'C:\Users\yexin\Desktop\liquid\ckpts\linear_regression_model_oil_final.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    
    with open(r'C:\Users\yexin\Desktop\liquid\ckpts\poly_features_oil_final.pkl', 'rb') as file:
        poly_features = pickle.load(file)
    
    
    containers = ['plastic']
    contents = ['oil']
    rotate_speed_list = [50, 100, 150]
    rotate_angle_list = [[87.4, 52.5, 178], [88.2, 35, 179.2]]
    grasp_height_list = [-10, -15]
    num_its = 10
    hold_time = 3
    p_duration = 20
    p1_duration = 11.5
    labels = 0
    final_weight = 127
    
    
    
    
        
        
    def equation(x, target = final_weight):
        
        # Convert x to a numpy array and ensure it's 2D with a single row
        x_reshaped = np.atleast_2d(x).reshape(1, -1)
        
        # Transform x to the same polynomial degree as used in training
        x_transformed = poly_features.transform(x_reshaped)
        
        # Predict using the transformed input
        return x + loaded_model.predict(x_transformed)[0] - target
    
    solution = fsolve(equation, 0, maxfev=100000)
    
    #lentils:4.1
    #rice:5.1
    #oil:2.6
    #water:3.1
    #vinegar:3
    final_weight = solution[0] - 2.6
    
    
    arm.set_gripper_speed(1000)
    create_dirs(current_dir)
    for container in containers:
        closed_pos, grip_height = set_height_and_pos(container)
        for content in contents:
            print('\ncontainer: '+container+' \ncontent: '+content+'\n')
            rotate_speed, rotate_angle, grasp_height, rotate_command, rotate_angle_command, grasp_height_command, fill_level = get_parameters(rotate_speed_list, rotate_angle_list, grasp_height_list)
            for iteration in range(num_its):     
                ready_for_pickup(arm)            
                input('\npress enter to begin pouring')
                data_1, data_2, data_3, data_4 = pick_up_pour(arm, closed_pos, grip_height, iteration, container, content, labels, hold_time, rotate_speed, rotate_angle, grasp_height, p_duration, p1_duration, final_weight)
                data = [data_1, data_2, data_3, data_4]
                with open(content+'_'+container+'_'+rotate_command+'_'+rotate_angle_command+'_'+grasp_height_command+'_'+fill_level+'_'+str(iteration)+'.pkl','wb') as file:
                    pickle.dump(data, file)
                #input('press enter when done pouring\n')
                #time.sleep(pouring_duration)
                #place_back(arm, grip_height, rotate_speed, grasp_height)
                if iteration != num_its-1:
                    input('please reset the containers and contents. \npress enter to continue')
                else:
                    resting_pos(arm)
        labels +=1
trial()


