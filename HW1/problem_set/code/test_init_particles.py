from main import *

if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    X_bar_random = init_particles_random(num_particles, occupancy_map)
    X_bar_free = init_particles_freespace(num_particles, occupancy_map)
    assert len(X_bar_random) == num_particles
    assert len(X_bar_free) == num_particles

    print(f"Saving output to {args.output}/")
    visualize_map(occupancy_map)
    visualize_timestep(X_bar_random, "test_init_particles_random", args.output)
    visualize_timestep(X_bar_free, "test_init_particles_free", args.output)