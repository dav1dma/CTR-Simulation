import pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected.")
    raise SystemExit

controller = pygame.joystick.Joystick(0)
controller.init()

print()
print("Controller:", controller.get_name())
print()
print("Press L1, then R1.")
print("Press CTRL+C when finished.")
print()

try:
    while True:

        for event in pygame.event.get():

            if event.type == pygame.JOYBUTTONDOWN:

                print(
                    f"BUTTON PRESSED -> {event.button}"
                )

            elif event.type == pygame.JOYBUTTONUP:

                print(
                    f"BUTTON RELEASED -> {event.button}"
                )

except KeyboardInterrupt:
    print()
    print("Finished.")

finally:
    pygame.quit()