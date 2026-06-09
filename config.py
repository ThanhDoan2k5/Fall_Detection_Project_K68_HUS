# config.py
CONFIG = {
    'IP_CAMERA': "http://192.168.5.103:8080/video",
    'WINDOW_SIZE': 30,
    'VOTE_THRESH': 4,                
    'CONFIDENCE_FALL': 0.60,         
    'RATIO_FALL': 0.7,               
    'ANGLE_FALL': 38.0,              
    'MIN_KP_CONF': 0.3,
    'LYING_FRAMES_THRESH': 45,       
    'MOVE_WINDOW_THRESH': 0.20,      
    'ALARM_ANGLE_THRESH': 50.0,      
    'ALARM_RATIO_THRESH': 0.85,      
    'RESET_ANGLE_THRESH': 32.0,      
    'AUTO_RESET_TIME': 4.0,          
    'MAX_ANGULAR_VELOCITY': 25.0,    
    'CSV_LOG': 'test_results_v3.0_final.csv'
}