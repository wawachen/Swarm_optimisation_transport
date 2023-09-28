# mpiexec -np 1 python examples/transport_test.py --scenario-name="imbalance" --kp=15.0 && \
# mpiexec -np 1 python examples/transport_test.py --scenario-name="proposed" --kp=1.0 # && \
# mpiexec -np 5 python examples/transport_test.py --scenario-name="ring" --kp=15.0 && \
# mpiexec -np 5 python examples/transport_test.py --scenario-name="imbalance" --kp=5.0 && \
# mpiexec -np 5 python examples/transport_test.py --scenario-name="proposed" --kp=5.0 && \
# mpiexec -np 5 python examples/transport_test.py --scenario-name="ring" --kp=5.0 && \
# mpiexec -np 1 python examples/swarm_flock_laser.py --scenario-name="circle" && \
# mpiexec -np 5 python examples/swarm_flock_laser.py --scenario-name="square" && \
# mpiexec -np 5 python examples/swarm_flock_laser.py --scenario-name="peanut" && \
mpiexec -np 1 python examples/swarm_flock_camera.py
# mpiexec -np 5 python examples/swarm_flock_camera.py --scenario-name="square" && \
# mpiexec -np 5 python examples/swarm_flock_camera.py --scenario-name="peanut" 

