"""Reliability hysteresis independent of images, geometry and particles."""

class PrimarySelector:
    def __init__(self, arm_names, allow_primary_switch=False):
        self.arm_names = tuple(arm_names)
        self.allow_primary_switch = allow_primary_switch
        self.primary = None
        self.enter_streak = 0
        self.exit_streak = 0
        self.dwell = 0
        self.candidate = None

    def update(self, reliability):
        """Return the confirmed primary, or None for ordinary joint tracking.

        FPRT can request only the first arm; APRT can request either arm.
        Entry/switch needs two confirmations. Switching and exit confirmation
        wait for ten active frames; exit needs three consecutive confirmations.
        """
        if len(self.arm_names) != 2:
            return None
        first, second = self.arm_names
        requested = None
        if reliability[first] >= 0.55 and reliability[second] <= 0.35:
            requested = first
        elif (self.allow_primary_switch
              and reliability[second] >= 0.55 and reliability[first] <= 0.35):
            requested = second

        current = self.primary
        if current is not None:
            self.dwell += 1

        # The same confirmation logic handles entry and changing the primary.
        if requested is not None and requested != current:
            if self.candidate == requested:
                self.enter_streak += 1
            else:
                self.candidate = requested
                self.enter_streak = 1
            can_switch = current is None or self.dwell >= 10
            if can_switch and self.enter_streak >= 2:
                current = requested
                self.enter_streak = 0
                self.exit_streak = 0
                self.dwell = 0
        else:
            self.candidate = None
            self.enter_streak = 0
            if current is not None and self.dwell >= 10:
                other = second if current == first else first
                should_exit = reliability[current] < 0.40 or reliability[other] >= 0.55
                self.exit_streak = self.exit_streak + 1 if should_exit else 0
                if self.exit_streak >= 3:
                    current = None
                    self.exit_streak = 0

        self.primary = current
        return current

