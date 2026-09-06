/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: safety_sidecar_filter.c
 *
 * MATLAB Coder version            : 25.1
 * C/C++ source code generated on  : 06-Sep-2026 19:16:29
 */

/* Include Files */
#include "safety_sidecar_filter.h"
#include <math.h>

/* Function Definitions */
/*
 * SAFETY_SIDECAR_FILTER: Deterministic Control Barrier Function (CBF)
 *
 *  --- ARCHITECTURE OVERVIEW ---
 *  This function serves as the "Action Governor" in a Simplex Architecture.
 *  It acts as a safety net wrapped around the primary AI agent.
 *  While the Agent is treated as an untrusted "black box",
 *  this sidecar uses strictly deterministic, formal Newtonian physics to
 * protect the spacecraft.
 *
 *  It operates statelessly on a microsecond basis, intercepting the AI's
 * requested actions (u_nominal) and passing them through three distinct
 * survival filters:
 *  1. Altitude (Crash Prevention)
 *  2. Rotation (Pitch Control)
 *  3. Fuel (Bingo Fuel Prevention)
 *
 *  If the AI's request is safe, it passes through untouched. If the request is
 * lethal, the Sidecar vetoes it and injects an emergency survival command.
 *
 * Arguments    : const double x[8]
 *                const double u_nominal[2]
 *                double u_actual[2]
 *                boolean_T *VetoTriggered
 *                double *h_alt
 *                double *h_fuel
 * Return Type  : void
 */
void safety_sidecar_filter(const double x[8], const double u_nominal[2],
                           double u_actual[2], boolean_T *VetoTriggered,
                           double *h_alt, double *h_fuel)
{
  double a_max;
  double b_x;
  double blending_zone;
  double d;
  double drop_slew;
  double dy_after;
  double g_apparent;
  double h_alt_now_tmp_tmp;
  double m_total;
  double margin;
  double t_slew;
  double t_slew_tmp;
  double theta;
  int y;
  /*  --- 1. UNPACK STATE & PARAMETERS --- */
  /*  The sidecar evaluates the exact physical reality of the craft at this
   * exact millisecond. */
  /*  Horizontal velocity */
  /*  Current altitude (meters) */
  /*  Current vertical velocity (m/s). Negative means falling. */
  theta = atan2(sin(x[4]), cos(x[4]));
  /*  Current pitch angle strictly wrapped to [-pi, pi] */
  /*  Current angular velocity (rad/s) */
  /*  Current main fuel mass (kg) */
  /*  Current RCS propellant mass (kg) */
  /*  Recalculate total mass (Mass dynamically changes as fuel is burned) */
  m_total = (x[6] + 4280.0) + x[7];
  /*  Unpack engine and physics constants */
  /*  --- 1b. HIGHER FIDELITY PHYSICS --- */
  /*  Calculate Centrifugal Lift (orbital mechanics) */
  g_apparent = fmax(0.0, 1.62 - x[2] * x[2] / (x[1] + 1.7374E+6));
  /*  Apparent gravity is reduced by orbital velocity */
  /*  Initialize the output to assume the AI is safe, until proven otherwise. */
  u_actual[1] = u_nominal[1];
  *VetoTriggered = false;
  /*  --- 1c. ATTITUDE RECOVERY COST (shared by every barrier below) --- */
  /*  Three separate barriers need to know what recovering from the current tilt
   * costs, */
  /*  so it is computed ONCE here. Previously each barrier asked "can it brake
   * while */
  /*  tilted like this" via T_max*cos(theta), which goes NEGATIVE past 90
   * degrees - so the */
  /*  altitude barrier commanded full thrust while inverted, and the fuel
   * barrier declared */
  /*  that no amount of propellant could arrest the descent. Both fired
   * constantly during */
  /*  a powered descent, which is flown deliberately near 90 degrees. */
  /*  */
  /*  The right question is what it costs to RECOVER and then brake: slewing
   * upright takes */
  /*  a bang-bang 2*sqrt(theta/alpha), during which the engine cannot fight
   * gravity, so */
  /*  the vehicle loses altitude and gains descent rate. */
  t_slew_tmp = fabs(theta);
  t_slew =
      2.0 *
      sqrt(fmax(t_slew_tmp, 1.0E-6) /
           (2000.0 / fmax((x[6] + x[7]) / 8500.0 * 45000.0 + 24000.0, 1.0)));
  a_max = fabs(x[3]);
  drop_slew = a_max * t_slew + 0.81 * (t_slew * t_slew);
  /*  altitude spent recovering */
  dy_after = a_max + 1.62 * t_slew;
  /*  descent rate once recovered */
  /*  --- 2. ALTITUDE BARRIER (CRASH PREVENTION) --- */
  /*  "The Action Governor shall assume control authority from the Primary */
  /*  AI Agent when the current altitude is less than or equal to d_stop +
   * safety_buffer_alt." */
  /*  The safety buffer defines a "Hover Height" above the actual ground (y=0).
   */
  /*  This ensures the sidecar catches the ship and stabilizes it in the air, */
  /*  rather than trying to stop exactly at the millimeter the landing gear hits
   * the dirt. */
  /*  Target hover height (meters) */
  b_x = 0.0;
  /*  Default to 0 required emergency thrust */
  /*  How much physical distance exists between the ship and the hover floor? */
  /*  MARGIN SENTINELS ARE FINITE ON PURPOSE. */
  /*  These two defaults used to be +Inf and -Inf. Both are only ever consumed
   * by */
  /*  comparisons, never by arithmetic, so realmax is exactly equivalent - and
   * it keeps */
  /*  IEEE special-case handling out of the generated flight code entirely. See
   */
  /*  codegen/README.md: with any Inf in this function, MATLAB Coder must emit
   */
  /*  rt_nonfinite.c, rtGetInf.c and rtGetNaN.c alongside the barrier, and the
   * artefact */
  /*  stops being straight-line arithmetic. Flight-software review generally
   * objects to */
  /*  Inf and NaN in the first place. */
  /*  not descending: no braking boundary applies */
  /*  cannot arrest at any throttle: always inside the wall */
  margin = 1.7976931348623157E+308;
  /*  Default: no braking boundary applies unless falling */
  blending_zone = 5.0;
  /*  Default warning envelope width (meters), rescaled below when falling */
  if (x[3] < -0.5) {
    /*  ACTIVE BRAKING: The ship is falling fast enough to warrant evaluation.
     */
    /*  Braking capability is evaluated for the RECOVERED attitude, not the
     * current one, */
    /*  with the cost of recovering charged separately. */
    /*  */
    /*  The original form was a_max = T_max*cos(theta)/m - g_apparent, which is
     * correct */
    /*  only while the vehicle stays tilted as it is. Past 90 degrees cos(theta)
     * goes */
    /*  NEGATIVE, so the barrier concluded the vehicle could not brake at all
     * and */
    /*  commanded full thrust - which, inverted, accelerates it toward the
     * ground. That */
    /*  never mattered while a separate rule capped tilt at 45 degrees, and
     * became live */
    /*  the moment a powered descent needed to point retrograde. Measured: the
     * barrier */
    /*  forced full thrust at 110 degrees of tilt at 15 km. */
    /*  */
    /*  The honest question is not "can it stop while tilted like this" but "can
     * it stop */
    /*  after recovering", so: charge the altitude lost during the slew, then
     * brake */
    /*  upright with the speed that slew leaves behind. */
    a_max = 45040.0 / m_total - g_apparent;
    /*  upright, so always the true authority */
    if (a_max > 0.0) {
      /*  Distance to stop = altitude spent recovering attitude, then the
       * braking */
      /*  distance at the speed that recovery leaves the vehicle carrying. */
      blending_zone = drop_slew + dy_after * dy_after / (2.0 * a_max);
      /*  Margin is the "slack" in the system. */
      /*  If margin == 0, the ship is at the exact point of no return. */
      margin = (x[1] - 1.5) - blending_zone;
      blending_zone = fmax(5.0, 0.3 * blending_zone);
    } else {
      /*  CRITICAL SCENARIO: Gravity is currently stronger than the available
       * vertical thrust capability. */
      /*  This happens if the ship is tilted too far (e.g. 90 degrees), or if
       * the engine is too weak. */
      /*  It is impossible to stop falling under these conditions, so the
       * vehicle is */
      /*  unconditionally inside the barrier. margin is only compared, never
       * used in */
      /*  arithmetic, so the finite sentinel is exact. */
      margin = -1.7976931348623157E+308;
    }
    /*  The Blending Zone (set above) prevents violent, structural-damaging
     * binary */
    /*  switching. Instead of waiting until margin == 0 and slamming the
     * throttle from */
    /*  0% to 100%, the sidecar smoothly ramps up its authority as the boundary
     * nears. */
    /*  */
    /*  The zone is scaled to the ship's actual braking distance rather than
     * being a */
    /*  fixed width. A fixed 50 m window is catastrophic at low speed: on a
     * gentle */
    /*  2 m/s descent the braking distance is ~1 m, so a 50 m window means the
     * sidecar */
    /*  holds authority continuously below 51 m altitude and the agent never
     * flies the */
    /*  approach itself. Scaling with d_min_stop keeps the barrier
     * proportionate: */
    /*  wide during a fast 25 m/s descent, narrow during a slow terminal hover.
     */
    if (margin <= 0.0) {
      /*  POINT OF NO RETURN: Spacecraft has pierced the mathematical boundary.
       */
      /*  Absolute maximum panic effort. The AI is entirely locked out of the
       * throttle. */
      b_x = 45040.0;
    } else if (margin < blending_zone) {
      /*  BLENDING ZONE: The ship is inside the warning envelope.  */
      /*  Ramp up minimum thrust smoothly as it approaches margin == 0. */
      b_x = 45040.0 * (1.0 - margin / blending_zone);
    }
  } else if (x[1] - 1.5 <= 0.5) {
    /*  HOVER MODE: Spacecraft has arrived at the safety buffer and arrested its
     * fall. */
    /*  To prevent bouncing, output exactly enough thrust to counteract apparent
     * gravity (F = mg). */
    b_x = m_total * g_apparent;
  }
  /*  --- 3. ROTATIONAL CONTROL BARRIER FUNCTION (ACTION GOVERNOR) --- */
  /*  "The RCS side thrusters shall be seized by the action governor to force
   * theta to 0 if the ship exceeds safe bounds, OR if it is in the altitude
   * danger zone." */
  /*  --- THE TILT LIMIT IS EARNED, NOT FIXED --- */
  /*  A 45 degree ceiling is correct for a terminal descent and makes a powered
   * descent */
  /*  impossible: braking off orbital velocity requires pointing the engine
   * retrograde, a */
  /*  pitch approaching 90 degrees, sustained for minutes. Measured on identical
   * initial */
  /*  conditions from 15.2 km and 1697 m/s: unguarded the vehicle lands at 0.29
   * m/s, and */
  /*  with a fixed 45 degree envelope it never gets below 12.5 km and is still
   * doing */
  /*  549 m/s when the clock expires. */
  /*  */
  /*  The barrier therefore decides the limit from PHYSICS IT MEASURES ITSELF,
   * rather than */
  /*  from a flight phase the controller declares. That distinction matters: a
   * barrier */
  /*  that trusts the component it exists to police is not independent of it,
   * and mode */
  /*  confusion in the nominal controller would silently widen the safety
   * envelope. */
  /*  */
  /*  The question a tilt limit is really asking is not "how far over is it" but
   * "can the */
  /*  recovery be afforded". Recovering from tilt theta means slewing back
   * upright, which */
  /*  the RCS does at alpha = Tau_max / I in a bang-bang time of
   * 2*sqrt(theta/alpha). */
  /*  Through that slew the engine cannot arrest the descent, so the vehicle
   * loses */
  /*  */
  /*      |dy| * t_rec + 0.5 * g * t_rec^2 */
  /*  */
  /*  of altitude. If h_alt - the margin beyond the stopping distance, already
   * computed */
  /*  above - covers that loss with margin, the tilt is recoverable and
   * permitted. Near */
  /*  the ground it never does, so the terminal limit re-emerges on its own
   * rather than */
  /*  being special-cased. */
  /*  require twice the altitude the slew actually costs */
  /*  Altitude margin beyond the stopping distance, evaluated UPRIGHT: the
   * question is */
  /*  what the vehicle could do once recovered, not what it can do while still
   * tilted. */
  h_alt_now_tmp_tmp = 45040.0 / m_total - g_apparent;
  /*  Hard structural ceiling, enforced whatever the altitude margin says. Past
   * this the */
  /*  vehicle is tumbling rather than manoeuvring. */
  /*  ~140 degrees */
  d = x[1] - x[3] * x[3] / (2.0 * fmax(0.1, h_alt_now_tmp_tmp));
  if (d > 2.0 * drop_slew) {
    a_max = 2.44;
  } else {
    a_max = 0.78539816339744828;
    /*  45 degrees - the terminal-descent envelope */
  }
  /*  ~20 degrees - tighter limit while inside the braking envelope */
  /*  The barrier fires in two cases: */
  /*    1. The ship exceeds 45 degrees of tilt anywhere in the flight envelope.
   * Past this */
  /*       point cos(theta) has eaten enough of the main engine's vertical
   * component that */
  /*       recovery authority is genuinely at risk. */
  /*    2. The ship is inside the altitude braking envelope AND tilted past 20
   * degrees, */
  /*       where wasted vertical thrust directly threatens the stopping
   * distance. */
  /*  */
  /*  It deliberately does NOT fire merely because the ship is in the braking
   * envelope. */
  /*  Seizing the RCS for the whole terminal descent pins theta at 0, and since
   */
  /*  ddx = -T*sin(theta)/m, that freezes horizontal velocity at whatever it was
   * when the */
  /*  barrier engaged. With a touchdown limit of 0.5 m/s lateral and approach
   * drift of */
  /*  10-20 m/s, that made a safe landing physically unreachable - the agent was
   * being */
  /*  asked to null drift with the only actuator that can do it taken away. The
   * agent now */
  /*  keeps torque authority to fly the approach, and the barrier intervenes
   * only when */
  /*  attitude itself becomes the hazard. */
  if ((t_slew_tmp > a_max) ||
      ((x[3] < -0.5) && (margin < blending_zone) && (t_slew_tmp > 0.35))) {
    /*  High-gain PD controller to aggressively torque the ship to vertical */
    /*  Gains anchored to the TERMINAL envelope, not the currently permitted
     * one. Tying */
    /*  them to a variable max_pitch would weaken the recovery torque precisely
     * when a */
    /*  wide envelope had been granted. */
    /*  Seize control of the RCS thrusters (u_actual(2)) */
    u_actual[1] = fmax(
        -2000.0,
        fmin(-2546.4790894703256 * theta - 2546.4790894703256 * x[5], 2000.0));
  }
  /*  --- 4. FUEL BARRIER (BINGO FUEL) --- */
  /*  "The System shall continuously calculate the emergency fuel reserve
   * required to arrest the current vertical velocity and  */
  /*  maintain a 1.0g hover for a duration of at least 3.0 seconds." */
  /*  3 seconds of emergency hover fuel */
  /*  Thrust required to hover */
  /*  Calculate fuel burn rate during hover (Linear scaling based on max flow
   * rate) */
  a_max = 15.6 * (m_total * g_apparent / 45040.0) * 3.0;
  y = 0;
  blending_zone = 0.0;
  if (x[3] < 0.0) {
    /*  Evaluated UPRIGHT, with the recovery charged separately - the same
     * correction */
    /*  the altitude barrier needed, and for the same reason. Using cos(theta)
     * here made */
    /*  a_max negative past 90 degrees, so t_stop became infinite and the
     * bingo-fuel */
    /*  barrier concluded no amount of propellant could ever arrest the descent.
     * During */
    /*  a braking burn, which is flown deliberately near 90 degrees, that fired
     */
    /*  constantly on a vehicle with 8 tonnes of usable propellant aboard. */
    /*  Unlike the margin sentinels above, this one cannot simply be made
     * finite: the */
    /*  old t_stop = Inf was MULTIPLIED by mdot, and realmax * mdot overflows
     * straight */
    /*  back to Inf. So the "cannot arrest" case sets the fuel DEMAND directly
     * instead */
    /*  of routing an infinite time through a multiplication. */
    if (h_alt_now_tmp_tmp <= 0.0) {
      /*  Genuinely cannot overcome gravity even pointed straight up, so no
       * quantity */
      /*  of propellant arrests this descent and the bingo condition is */
      /*  unconditionally true. Any demand exceeding a full tank expresses that
       */
      /*  exactly, because m_main_fuel can never exceed max_main_fuel. */
      blending_zone = 8201.0;
    } else {
      /*  Time to recover attitude, then to stop at the speed recovery leaves
       * behind. */
      /*  Calculate EXACTLY how much fuel will be consumed executing that stop
       */
      blending_zone = 15.6 * (t_slew + dy_after / h_alt_now_tmp_tmp);
    }
    /*  "Upon reaching bingo fuel level, the Action Governor shall immediately
     * force a maximum-thrust suicide burn" */
    /*  If the tank level drops to the exact amount of fuel required to stop +
     * the 3-second reserve... */
    if (x[6] <= blending_zone + a_max) {
      /*  Force the AI into a "Suicide Burn" to land the ship NOW before it
       * physically runs out of gas. */
      y = 45040;
      /*  BUG FIX: We DO NOT zero out the torque here. The ship might be tilted!
       */
      /*  We must allow the Rotational CBF to continue fighting to right the
       * ship. */
    }
  }
  /*  --- 4b. RCS FUEL BARRIER --- */
  /*  If the sidecar or the AI depletes the RCS fuel, it is physically
   * impossible to output torque. */
  if (x[7] <= 0.0) {
    u_actual[1] = 0.0;
    /*  Override AI and Rotational CBF */
  }
  /*  --- 5. ACTION FILTER & HARDWARE CLAMPS --- */
  /*  Physics Floor: What is the absolute minimum thrust needed to survive this
   * millisecond? */
  /*  The most restrictive requirement between the Altitude CBF and the Fuel CBF
   * is taken. */
  /*  Hardware Ceiling: The system cannot physically fire harder than the
   * engine's mechanical limit. */
  /*  Ensure the Sidecar obeys the laws of physics: Do not allow the safety net
   * to demand more thrust than exists. */
  /*  bare minimum threshold to maintain safe flight  */
  /*  Combine everything: Allow the AI to command whatever it wants, AS LONG AS
   * it is bounded  */
  /*  between the Survival Floor (T_lower_bound) and the Hardware Ceiling
   * (T_upper_bound). */
  u_actual[0] = fmax(fmin(fmax(b_x, y), 45040.0), fmin(u_nominal[0], 45040.0));
  /*  --- 6. LOGGING & STATE AUGMENTATION --- */
  /*  Check if the Sidecar had to MATERIALLY alter the AI's requested command.
   */
  /*  */
  /*  The tolerance is a fraction of actuator authority, not an absolute newton
   * count. A */
  /*  0.1 N threshold on a 45 kN engine is a rounding error: it counted
   * sub-newton */
  /*  differences as safety interventions, so an agent flying along the thrust
   * floor */
  /*  registered dozens of "engagements" per descent. That inflates the veto
   * metric the */
  /*  A/B study reports on and makes the barrier look far twitchier than it is.
   */
  /*  ~225 N of 45 kN */
  /*  ~10 Nm of 2 kNm */
  if ((fabs(u_actual[0] - u_nominal[0]) > 225.20000000000002) ||
      (fabs(u_actual[1] - u_nominal[1]) > 10.0)) {
    /*  This flag is sent back to the environment. The AI receives a large
     * penalty  */
    /*  every time this triggers. This teaches the AI to fear the boundaries and
     * learn  */
    /*  to fly so perfectly that the Sidecar never has to wake up. */
    *VetoTriggered = true;
  }
  /*  Export the continuous barrier values (h).  */
  /*  By feeding these values directly into the neural network's observation
   * state,  */
  /*  the AI is given "eyes" to mathematically see the invisible boundaries
   * approaching. */
  *h_alt = d;
  *h_fuel = x[6] - (blending_zone + a_max);
}

/*
 * File trailer for safety_sidecar_filter.c
 *
 * [EOF]
 */
