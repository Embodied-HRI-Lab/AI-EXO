#ifndef MOTOR_CONTROL_TMOTOR_H
#define MOTOR_CONTROL_TMOTOR_H

#include <Arduino.h>
#include <FlexCAN_T4.h>
#include <stdint.h>

#ifndef __IMXRT1062__
#error "This driver requires a Teensy 4.x / i.MXRT1062 target."
#endif

/*
 * Driver for 自定义 CAN 通信协议 V3.07b0
 *
 * Normal host command:
 *   StdID = 0x100 | Dev_addr
 *   Data[0] = command
 *   Data[1..] = payload
 *
 * Motor reply:
 *   StdID = Dev_addr
 *
 * MIT command:
 *   StdID = 0x500 | Dev_addr
 *   DLC = 8
 *   No command byte in payload.
 *
 * Normal multi-byte values use little-endian byte order.
 * MIT packed fields follow the explicit byte/bit layout in the manual.
 */
class Motor_Control_Tmotor
{
public:
    // One shared CAN3 controller for all motor objects.
    static FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16> Can3;

    explicit Motor_Control_Tmotor(uint8_t id, int can_id_unused = 0);
    ~Motor_Control_Tmotor() = default;

    // CAN
    void initial_CAN(uint32_t baud = 1000000);
    bool send_CAN_message();
    bool handle_reply(const CAN_message_t &msgR);

    // Legacy-compatible receive wrappers
    void unpack_reply(CAN_message_t msgR, float initial_pos = 0.0f);
    void unpack_pos_vel(CAN_message_t msgR, float initial_pos = 0.0f);
    void unpack_torque(CAN_message_t msgR);

    // System/read commands
    bool reboot();                    // 0x00
    bool request_versions();          // 0xA0
    bool request_q_current();         // 0xA1
    bool request_speed();             // 0xA2
    bool request_position();          // 0xA3
    bool request_realtime();          // 0xA4
    bool request_status();            // 0xAE
    bool error_clear();               // 0xAF

    // Legacy convenience wrappers
    bool request_pos_vel();           // sends A3 then A2
    bool request_torque();            // sends A1

    // Parameter commands
    bool request_motor_parameters();                // 0xB0
    bool set_origin();                              // 0xB1
    bool set_position_max_speed_rpm(float rpm);     // 0xB2
    bool set_max_q_current_A(float amp);            // 0xB3
    bool set_q_current_slope_Aps(float amp_per_s); // 0xB4
    bool set_speed_accel_rpmps(float rpm_per_s);   // 0xB5

    bool request_position_kp();                     // 0xB6
    bool set_position_kp(float kp);                 // 0xB6
    bool request_position_ki();                     // 0xB7
    bool set_position_ki(float ki);                 // 0xB7
    bool request_speed_kp();                        // 0xB8
    bool set_speed_kp(float kp);                    // 0xB8
    bool request_speed_ki();                        // 0xB9
    bool set_speed_ki(float ki);                    // 0xB9

    // Normal control
    bool command_q_current_A(float current_A);      // 0xC0
    bool command_speed_rpm(float rpm);              // 0xC1
    bool command_abs_position_count(int32_t count); // 0xC2
    bool command_rel_position_count(int32_t count); // 0xC3
    bool command_abs_position_rad(float rad);       // helper -> 0xC2
    bool command_rel_position_rad(float rad);       // helper -> 0xC3
    bool return_to_origin();                        // 0xC4
    bool set_brake(bool closed);                    // 0xCE
    bool request_brake_state();                     // 0xCE, payload 0xFF
    bool disable_motor();                           // 0xCF

    // Nm -> Iq using torque constant from B0 or set_torque_constant().
    // This function DOES NOT automatically multiply/divide by gear_ratio.
    bool command_torque_Nm(float torque_Nm);

    // MIT mode
    bool request_mit_limits();                      // 0xF0
    bool set_mit_limits(float pos_max_rad,
                        float vel_max_rad_s,
                        float torque_max_Nm);       // 0xF0
    bool request_mit_state();                       // 0xF1
    bool mit_ctl_cmd(float p_des,
                         float v_des,
                         float kp,
                         float kd,
                         float t_ff);

    // Legacy-name compatibility with the old motor API.
    // New protocol does not need separate normal-mode start commands.
    bool enter_control_mode();
    bool exit_control_mode();
    bool motor_reset();
    bool encoder_reset();
    bool motor_start();
    bool motor_end();
    bool torque_ctl_mode_start();
    bool speed_ctl_mode_start();
    bool mit_ctl_mode_start();
    bool speed_cmd(float rpm);
    bool speed_cmd();              // safe compatibility overload -> 0 rpm
    bool torque_cmd(float tau_Nm);

    // Conversion helpers for MIT packed fields
    int float_to_uint(float x,
                      float x_min,
                      float x_max,
                      uint8_t nbits) const;

    float uint_to_float(int x_int,
                        float x_min,
                        float x_max,
                        uint8_t nbits) const;

    // Configuration helpers
    void set_device_id(uint8_t id);
    uint8_t get_device_id() const { return ID; }

    void set_software_zero_rad(float raw_zero_rad);
    void set_software_zero_here();

    void set_torque_constant(float kt);
    bool torque_constant_is_valid() const { return torque_constant_valid; }

    void set_software_current_limit_A(float limit_A);
    void set_software_torque_limit_Nm(float limit_Nm);

    // ---------------------------------------------------------------------
    // Public state
    // ---------------------------------------------------------------------
    // Old code compatibility
    float pos;       // rad, software-zeroed
    float spe;       // rad/s
    float torque;    // estimated/feedback Nm
    float temp;      // degC

    // Explicit feedback values
    float q_current_A;
    float speed_rpm;
    float single_turn_pos_rad;
    float raw_position_rad;
    float bus_voltage_V;
    float bus_current_A;

    uint8_t operation_mode;
    uint8_t fault_code;
    uint8_t mit_status;
    uint8_t brake_state;

    // B0 motor parameters
    uint8_t pole_pairs;
    float torque_constant;
    uint8_t gear_ratio;
    bool torque_constant_valid;

    // B6-B9 values
    float position_kp;
    float position_ki;
    float speed_kp;
    float speed_ki;

    // F0 MIT ranges
    float MIT_P_MAX;     // rad
    float MIT_V_MAX;     // rad/s
    float MIT_T_MAX;     // Nm
    bool mit_limits_valid;

    bool last_send_ok;

private:
    CAN_message_t msgW;
    uint8_t ID;

    // Software zero / unwrap state
    float software_zero_rad;
    bool have_single_turn_sample;
    uint16_t last_single_turn_count;
    int64_t unwrapped_count;

    // Conservative software clamps; user can change them.
    float software_current_limit_A;
    float software_torque_limit_Nm;

    // Generic TX helpers
    bool send_command(uint8_t command,
                      const uint8_t *payload = nullptr,
                      uint8_t payload_len = 0);
    bool send_u32_command(uint8_t command, uint32_t value);
    bool send_i32_command(uint8_t command, int32_t value);
    bool send_float_command(uint8_t command, float value);

    // Parsers
    void parse_A0(const CAN_message_t &msg);
    void parse_A1_or_C0(const CAN_message_t &msg);
    void parse_A2_or_C1(const CAN_message_t &msg);
    void parse_A3_family(const CAN_message_t &msg);
    void parse_A4(const CAN_message_t &msg);
    void parse_AE_or_CF(const CAN_message_t &msg);
    void parse_AF(const CAN_message_t &msg);
    void parse_B0(const CAN_message_t &msg);
    void parse_B1(const CAN_message_t &msg);
    void parse_B6_B9(const CAN_message_t &msg);
    void parse_CE(const CAN_message_t &msg);
    void parse_F0(const CAN_message_t &msg);
    void parse_F1(const CAN_message_t &msg);

    void update_position_from_single_turn(uint16_t count);
    void update_position_from_multiturn(uint16_t single_count, int32_t total_count);

    // Byte helpers
    static float clampf(float x, float lo, float hi);

    static void write_u16_le(uint8_t *dst, uint16_t value);
    static void write_u32_le(uint8_t *dst, uint32_t value);
    static void write_i32_le(uint8_t *dst, int32_t value);
    static void write_f32_le(uint8_t *dst, float value);

    static uint16_t read_u16_le(const uint8_t *src);
    static int16_t read_i16_le(const uint8_t *src);
    static uint32_t read_u32_le(const uint8_t *src);
    static int32_t read_i32_le(const uint8_t *src);
    static float read_f32_le(const uint8_t *src);
};

#endif // MOTOR_CONTROL_TMOTOR_H