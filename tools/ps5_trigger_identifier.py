import pygame
import time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected.")
    raise SystemExit

controller = pygame.joystick.Joystick(0)
controller.init()

print()
print("PS5 TRIGGER TEST")
print("----------------")
print("Leave both triggers released.")
print("Then press L2 slowly, release it.")
print("Then press R2 slowly, release it.")
print("Press CTRL+C when finished.")
print()

try:
    while True:

        pygame.event.pump()

        axis_4 = controller.get_axis(4)
        axis_5 = controller.get_axis(5)

        print(
            f"Axis 4 = {axis_4:+.3f}    "
            f"Axis 5 = {axis_5:+.3f}"
        )

        time.sleep(0.2)

except KeyboardInterrupt:

    print()
    print("Finished.")

finally:

    pygame.quit()