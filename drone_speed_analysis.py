import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import time
from collections import deque
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
import yaml

class SpeedEstimator:
    def __init__(self, frame_width: int, frame_height: int, 
                 real_width: float = 10.0, real_height: float = 10.0):
        
     
        self.pixel_to_meter = min(real_width/frame_width, real_height/frame_height)
        self.speed_filter = SpeedFilter()
        
    def calculate_speed(self, p1: np.ndarray, p2: np.ndarray, dt: float) -> float:
     
        distance_pixels = np.sqrt(
            (p2[0] - p1[0])**2 + 
            (p2[1] - p1[1])**2
        )
        
        distance = distance_pixels * self.pixel_to_meter
        
        if dt > 0:
            speed = distance / dt
            return self.speed_filter.update(speed)
        return 0.0

class SpeedFilter:
    def __init__(self, window_size: int = 5):
        self.speeds = deque(maxlen=window_size)
        
    def update(self, speed: float) -> float:
        self.speeds.append(speed)
        valid_speeds = [s for s in self.speeds 
                       if abs(s - np.mean(self.speeds)) < 2*np.std(self.speeds)]
        return np.mean(valid_speeds) if valid_speeds else speed

class DroneTracker:
    def __init__(self, max_age: int = 5):
        self.tracker = DeepSort(max_age=max_age)
        
    def update(self, detections: List, frame: np.ndarray):
        return self.tracker.update_tracks(detections, frame=frame)

class CoordinateFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32) * 0.03
        self.initialized = False

    def update(self, x: float, y: float) -> Tuple[float, float]:
        measurement = np.array([[x], [y]], np.float32)
        
        if not self.initialized:
            self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        
        return prediction[0, 0], prediction[1, 0]

class TrajectoryAnalyzer:
    def __init__(self, video_width: int, video_height: int):
        self.trajectories: Dict = {}
        self.speeds: Dict = {}
        self.directions: Dict = {}
        self.distances: Dict = {}
        self.bbox_sizes: Dict = {}
        self.coord_filters: Dict = {}
        self.speed_estimator = SpeedEstimator(video_width, video_height)
        self.start_time = time.time()
        self.video_width = video_width
        self.video_height = video_height
        
        self.focal_length = 1000
        self.real_drone_width = 0.35
        self.distance_filter = MovingAverageFilter(window_size=5)

    def estimate_distance(self, bbox_width: float) -> float:
        distance = (self.real_drone_width * self.focal_length) / bbox_width
        return self.distance_filter.update(distance)

    def smooth_trajectory(self, points: np.ndarray, window: int = 5) -> np.ndarray:
        if len(points) < window:
            return points
            
        smoothed = np.zeros_like(points, dtype=np.float32)
        smoothed[:, 0] = savgol_filter(points[:, 0], window, 3)
        smoothed[:, 1] = savgol_filter(points[:, 1], window, 3)
        return smoothed

    def update_trajectory(self, drone_id: int, x: float, y: float, 
                         bbox_width: float, timestamp: float) -> None:
        if drone_id not in self.trajectories:
            self.trajectories[drone_id] = []
            self.speeds[drone_id] = []
            self.directions[drone_id] = []
            self.distances[drone_id] = []
            self.bbox_sizes[drone_id] = []
            self.coord_filters[drone_id] = CoordinateFilter()
            
        filtered_x, filtered_y = self.coord_filters[drone_id].update(x, y)
        
        current_point = np.array([filtered_x, filtered_y])
        self.trajectories[drone_id].append((current_point, timestamp))
        
        self.bbox_sizes[drone_id].append(bbox_width)
        distance = self.estimate_distance(bbox_width)
        self.distances[drone_id].append(distance)
        
        if len(self.trajectories[drone_id]) >= 2:
            prev_point, prev_time = self.trajectories[drone_id][-2]
            curr_point, curr_time = self.trajectories[drone_id][-1]
            
            speed = self.speed_estimator.calculate_speed(
                prev_point, curr_point, curr_time - prev_time
            )
            self.speeds[drone_id].append(speed)
            
            angle = np.arctan2(
                curr_point[1] - prev_point[1],
                curr_point[0] - prev_point[0]
            )
            self.directions[drone_id].append(angle)

    def get_movement_analysis(self, drone_id: int) -> dict:
        if drone_id not in self.trajectories:
            return {}
            
        speeds = self.speeds[drone_id]
        distances = self.distances[drone_id]
        current_speed = speeds[-1] if speeds else 0
        current_distance = distances[-1] if distances else 0
        
        bbox_sizes = self.bbox_sizes[drone_id]
        if len(bbox_sizes) >= 2:
            size_change = bbox_sizes[-1] - bbox_sizes[-2]
            approaching = size_change > 0
        else:
            approaching = None
        
        return {
            'current_speed': current_speed,
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': max(speeds) if speeds else 0,
            'speed_kmh': current_speed * 3.6,
            'distance_to_camera': current_distance,
            'is_approaching': approaching,
            'main_direction': np.mean(self.directions[drone_id]) if self.directions[drone_id] else 0,
            'trajectory_points': len(self.trajectories[drone_id])
        }

    def plot_analysis(self) -> None:
        try:
            if not self.trajectories:
                return

            fig = plt.figure(figsize=(15, 10))
            
            ax1 = fig.add_subplot(221)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(self.trajectories)))
            
            first_trajectory = next(iter(self.trajectories.values()))
            if first_trajectory:
                points = np.array([p[0] for p in first_trajectory])
                frame_width = int(self.video_width)
                frame_height = int(self.video_height) 
            
            for (drone_id, trajectory), color in zip(self.trajectories.items(), colors):
                points = np.array([p[0] for p in trajectory])
                
                ax1.plot(points[:, 0], points[:, 1], 
                        color=color, label=f'Drone {drone_id}')
                ax1.scatter(points[0, 0], points[0, 1], 
                           color=color, marker='o')
                ax1.scatter(points[-1, 0], points[-1, 1], 
                           color=color, marker='s')
            
            ax1.set_xlim([0, frame_width])
            ax1.set_ylim([frame_height, 0])
            
            ax1.grid(True, alpha=0.3)
            x_ticks = np.linspace(0, frame_width, 11)  
            y_ticks = np.linspace(0, frame_height, 11) 
            
            ax1.set_xticks(x_ticks)
            ax1.set_yticks(y_ticks)
            
            ax1.set_xticklabels([f'{int(x)}' for x in x_ticks])
            ax1.set_yticklabels([f'{int(y)}' for y in y_ticks])
            
            ax1.set_title('Quỹ đạo')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            ax1.legend()

            ax1.set_aspect('equal')

            ax2 = fig.add_subplot(222)
            for drone_id, speeds in self.speeds.items():
                times = range(len(speeds))
                ax2.plot(times, speeds, label=f'Drone {drone_id}')

                window = 5
                if len(speeds) > window:
                    moving_avg = np.convolve(speeds, np.ones(window)/window, mode='valid')
                    ax2.plot(range(window-1, len(speeds)), moving_avg, '--', 
                            alpha=0.5, label=f'MA{window} Drone {drone_id}')
            
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Vận tốc theo thời gian')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Vận tốc (m/s)')
            ax2.legend()

            ax3 = fig.add_subplot(223)
            for drone_id, speeds in self.speeds.items():
                ax3.hist(speeds, bins=30, alpha=0.5, 
                        label=f'Drone {drone_id}')
            ax3.grid(True, alpha=0.3)
            ax3.set_title('Phân bố vận tốc')
            ax3.set_xlabel('Vận tốc (m/s)')
            ax3.set_ylabel('Tần suất')
            ax3.legend()

            ax4 = fig.add_subplot(224, projection='polar')
            for drone_id, directions in self.directions.items():
                ax4.hist(directions, bins=36, alpha=0.5, 
                        label=f'Drone {drone_id}')
            ax4.set_title('Phân bố hướng di chuyển')
            ax4.legend()

            plt.tight_layout()
            plt.savefig('drone_speed_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

        except Exception as e:
            print(f"Error: {str(e)}")
        
        finally:
            plt.close()

    def plot_3d_trajectory(self) -> None:
        if not self.trajectories:
            print("Không có trajectory")
            return

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.trajectories)))

        for (drone_id, trajectory), color in zip(self.trajectories.items(), colors):
            points = np.array([p[0] for p in trajectory])
            smoothed_points = self.smooth_trajectory(points)
            
            x = smoothed_points[:, 0]
            y = smoothed_points[:, 1]
            
            z = np.array(self.distances[drone_id])
            if len(z) > 5:
                z = savgol_filter(z, 5, 3)

            if len(x) > 3:
                t = np.arange(len(x))
                t_smooth = np.linspace(0, len(x)-1, 200)
                
                x_spline = make_interp_spline(t, x, k=3)(t_smooth)
                y_spline = make_interp_spline(t, y, k=3)(t_smooth)
                z_spline = make_interp_spline(t, z, k=3)(t_smooth)

                ax.plot3D(x_spline, y_spline, z_spline, color=color, 
                        label=f'Drone {drone_id}', linewidth=2)
            else:
                ax.plot3D(x, y, z, color=color, label=f'Drone {drone_id}')
            
            ax.scatter(x[0], y[0], z[0], color=color, marker='o', s=100, 
                    label=f'Start {drone_id}')
            ax.scatter(x[-1], y[-1], z[-1], color=color, marker='s', s=100, 
                    label=f'End {drone_id}')

            ax.text(x[-1], y[-1], z[-1], 
                    f'Drone {drone_id}\n'
                    f'v={self.speeds[drone_id][-1]:.1f}m/s\n'
                    f'd={z[-1]:.1f}m',
                    color=color)

        ax.set_xlim([0, 640])
        ax.set_ylim([0, 480])
        ax.set_zlim([0, 5.0])

        ax.set_xticks(np.arange(0, 641, 40))
        ax.set_yticks(np.arange(0, 481, 40))
        ax.set_zticks(np.arange(0, 2.1, 0.25))

        ax.set_xlabel('X (pixels)', labelpad=10)
        ax.set_ylabel('Y (pixels)', labelpad=10)
        ax.set_zlabel('Distance to camera (meters)', labelpad=10)
        ax.set_title('3D Drone Trajectory')

        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.view_init(elev=20, azim=45)

        xx, yy = np.meshgrid(np.linspace(0, 640, 2),
                            np.linspace(0, 480, 2))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

        ax.legend()
        plt.tight_layout()
        plt.savefig('drone_3d_trajectory.png', dpi=300, bbox_inches='tight')
        plt.show()

    def get_smoothed_trajectory(self, track_id: int, window: int = 30) -> np.ndarray:
        if track_id not in self.trajectories:
            return None
            
        points = [p[0] for p in self.trajectories[track_id][-window:]]
        if len(points) < 3:
            return np.array(points)
            
        points = np.array(points)
        return self.smooth_trajectory(points)

class VideoProcessor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.stream = None
        self.writer = None
        
    def initialize(self) -> Tuple[int, int, int]:
        self.stream = cv2.VideoCapture(self.input_path)

        width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.stream.get(cv2.CAP_PROP_FPS))

        self.writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (width, height)
        )
        
        if not self.writer.isOpened():
            raise IOError("Không thể tạo video output")
            
        return width, height, fps
        
    def release(self):
        if self.stream is not None:
            self.stream.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()

class MovingAverageFilter:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        
    def update(self, value: float) -> float:
        self.values.append(value)
        return np.mean(self.values)

def main():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {
            'input_path': "10.mp4",
            'output_path': "result_10.mp4",
            'conf_thresh': 0.45,
            'iou_thresh': 0.45,
            'img_size': (640, 640),
            'show_trajectory': True,
            'trajectory_length': 30,
            'trajectory_thickness': 2
        }
    
    input_path = config['input_path']
    output_path = config['output_path']
    conf_thresh = config['conf_thresh']
    iou_thresh = config['iou_thresh']
    img_size = config['img_size']

    video_proc = VideoProcessor(input_path, output_path)
    width, height, fps = video_proc.initialize()
    
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend('best.pt', device=device)
    model.eval()
    
    if device.type != 'cpu':
        model.half()

    stride = model.stride
    img_size = check_img_size(img_size, s=stride)
    model.warmup(imgsz=(1, 3, *img_size))

    drone_tracker = DroneTracker(max_age=5)
    analyzer = TrajectoryAnalyzer(width, height)

    cv2.namedWindow('Drone Tracking G11', cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = video_proc.stream.read()
            if not ret:
                break

            current_time = time.time() - analyzer.start_time

            img = letterbox(frame, img_size, stride=stride, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if device.type != 'cpu' else img.float()
            img /= 255
            if len(img.shape) == 3:
                img = img[None]

            #Detection
            with torch.no_grad():
                pred = model(img, augment=False, visualize=False)
            pred = non_max_suppression(
                pred, conf_thresh, iou_thresh, 
                None, False, max_det=1000
            )

            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], frame.shape
                    ).round()

                    detections = []
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = map(int, xyxy)
                        detections.append([
                            [x1, y1, x2-x1, y2-y1], 
                            float(conf), 
                            int(cls)
                        ])

                    tracks = drone_tracker.update(detections, frame)

                    for track in tracks:
                        if track.is_confirmed():
                            track_id = track.track_id
                            x1, y1, x2, y2 = map(int, track.to_ltrb())
                            
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            analyzer.update_trajectory(
                                track_id, center_x, center_y, x2-x1, current_time
                            )
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                        (0, 255, 0), 2)
                            
                            analysis = analyzer.get_movement_analysis(track_id)
                            if analysis:
                                info = f"ID:{track_id} "
                                info += f"v:{analysis['current_speed']:.1f}m/s "
                                info += f"({analysis['speed_kmh']:.1f}km/h)"
                                cv2.putText(frame, info, (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, (0, 255, 0), 2)

                            if track_id in analyzer.trajectories:
                                trajectory_points = [p[0] for p in analyzer.trajectories[track_id][-30:]]
                                if len(trajectory_points) >= 2:
                                    trajectory_array = np.array(trajectory_points, dtype=np.int32)
                                    
                                    points = trajectory_array.reshape((-1, 1, 2))
                                    for i in range(len(points)-1):
                                        color = (
                                            int(255 * (1 - i/len(points))),  
                                            0,                              
                                            int(255 * (i/len(points)))      
                                        )
                                        cv2.line(frame, 
                                                tuple(points[i][0]), 
                                                tuple(points[i+1][0]), 
                                                color, 2)
                                    
                                    cv2.circle(frame, 
                                             tuple(trajectory_array[0]), 
                                             5, (0, 0, 255), -1)  
                                    cv2.circle(frame, 
                                             tuple(trajectory_array[-1]), 
                                             5, (255, 0, 0), -1) 

            video_proc.writer.write(frame)
            cv2.imshow('Drone Tracking G11', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Lỗi: {str(e)}")
    finally:
        video_proc.release()
        torch.cuda.empty_cache()

    for drone_id in analyzer.trajectories.keys():
        analysis = analyzer.get_movement_analysis(drone_id)
        print(f"\nDrone {drone_id}:")
        print(f"- Vận tốc hiện tại: {analysis['current_speed']:.2f}m/s ({analysis['speed_kmh']:.2f}km/h)")
        print(f"- Số điểm quỹ đạo: {analysis['trajectory_points']}")

    try:
        analyzer.plot_analysis()
        analyzer.plot_3d_trajectory()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 