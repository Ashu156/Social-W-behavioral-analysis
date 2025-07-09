%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Protocol: Dual Linear Track Alternation with Hold and Barrier Delay
% Author: [Your Name]
% Description:
%   - Two rats run independently on separate linear tracks.
%   - Rats must alternate between two reward wells to receive rewards.
%   - Hold times (nose-pokes) and motorized barriers are used to shape behavior.
%   - Used as pre-training for W-track spatial alternation/observational task.
%
% Key Hardware Ports:
%   Reward Pumps: Track12 -> Pump1 (port 3), Pump2 (port 1)
%                 TrackAB -> PumpA (port 5), PumpB (port 8)
%   IR Inputs:    Well1 = portin[15], Well2 = portin[10]
%                 WellA = portin[2],  WellB = portin[4]
%   Barriers:     Define b1, b2, bA, bB in ECU config or hardware file
%
% Parameters:
%   deliverPeriod1: Duration (ms) to hold reward port open
%   holdTime: Required time (ms) for nose-poke before reward
%   barrier_down_delay: Time (ms) to keep barrier up after reward
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Adjustable Parameters
int deliverPeriod1 = 600        % Reward duration (ms) – adjust based on pump
int holdTime = 200             % Hold time requirement (ms)
int barrier_down_delay = 20000   % Barrier delay after reward (ms)

% Tracking and Task State Variables
int lastWell12 = 3              % Track12 – last rewarded well (init to invalid)
int lastWellAB = 3              % TrackAB – last rewarded well (init to invalid)
int count12 = 0                 % Track12 – reward count
int countAB = 0                 % TrackAB – reward count
int lockout12 = 0               % Track12 – poke lockout state
int lockoutAB = 0               % TrackAB – poke lockout state
int currWell12 = 0             % Track12 – current well identity
int currWellAB = 0             % TrackAB – current well identity
int lockout_enabled = 1        % Dummy control variable (always on)

% Output Ports – Reward Pumps
int rewardPump1 = 3             % Track12 – Well 1 Pump (port 3)
int rewardPump2 = 1             % Track12 – Well 2 Pump (port 1)
int rewardPumpA = 5             % TrackAB – Well A Pump (port 5)
int rewardPumpB = 8             % TrackAB – Well B Pump (port 8)
int rewardWell12 = 1 % the pump to actuate (will be assigned rewardPump)
int rewardWellAB = 1
% Output Ports – Barrier Definitions (define b1, b2, etc. externally)
int barrier1 = 17
int barrier2 = 18
int barrierA = 22
int barrierB = 20

int active_barrier12 = 0
int active_barrierAB = 0

% switch off the barriers

% portout[barrier1] = 1
% portout[barrier2] = 1
% portout[barrierA] = 1
% portout[barrierB] = 1

updates off 16

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reward Delivery Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function 1 % Deliver reward on Track12
    portout[rewardWell12] = 1
    do in deliverPeriod1
        portout[rewardWell12] = 0
    end
end;

function 2 % Deliver reward on TrackAB
    portout[rewardWellAB] = 1
    do in deliverPeriod1
        portout[rewardWellAB] = 0
    end
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Barrier Control Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Track12 Barriers
function 3 % Raise active barrier
    portout[active_barrier12] = 1
    if (active_barrier12 == barrier1) do
        disp('Barrier 1 UP')
    else do
        disp('Barrier 2 UP')
    end
end;

function 4 % Lower active barrier after delay
    do in barrier_down_delay
        portout[active_barrier12] = 0
        if (active_barrier12 == barrier1) do
            disp('Barrier 1 Down')
        else do
            disp('Barrier 2 Down')
        end
    end
end;

% TrackAB Barriers
function 5
    portout[active_barrierAB] = 1
    if (active_barrierAB == barrierA) do
        disp('Barrier A UP')
    else do
        disp('Barrier B UP')
    end
end;

function 6
    do in barrier_down_delay
        portout[active_barrierAB] = 0
        if (active_barrierAB == barrierA) do
            disp('Barrier A Down')
        else do
            disp('Barrier B Down')
        end
    end
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Callback Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% === TRACK 12 ===

callback portin[15] up % Well 1
    disp('Poke in well1')
    currWell12 = 1
    if (lockout12 == 0) do
        lockout12 = 1
        if (lastWell12 != 1) do
            active_barrier12 = barrier1
            trigger(3)
            if (currWell12 == 1) do in holdTime
                disp('Rewarding well 1')
                rewardWell12 = rewardPump1
                trigger(1)
                lastWell12 = 1
                count12 = count12 + 1
                disp(count12)
                trigger(4)
            end
        end
        if (lockout_enabled == 1) do in holdTime
            lockout12 = 0
        end
    end
end;

callback portin[15] down
    disp('Unpoke well1')
    currWell12 = 0
end;

callback portin[10] up % Well 2
    disp('Poke in well2')
    currWell12 = 2
    if (lockout12 == 0) do
        lockout12 = 1
        if (lastWell12 != 2) do
            active_barrier12 = barrier2
            trigger(3)
            if (currWell12 == 2) do in holdTime
                disp('Rewarding well 2')
                rewardWell12 = rewardPump2
                trigger(1)
                lastWell12 = 2
                count12 = count12 + 1
                disp(count12)
                trigger(4)
            end
        end
        if (lockout_enabled == 1) do in holdTime
            lockout12 = 0
        end
    end
end;

callback portin[10] down
    disp('Unpoke well2')
    currWell12 = 0
end;

% === TRACK AB ===

callback portin[8] up % Well A
    disp('Poke in wellA')
    currWellAB = 1
    if (lockoutAB == 0) do
        lockoutAB = 1
        if (lastWellAB != 1) do
            active_barrierAB = barrierA
            trigger(5)
            if (currWellAB == 1) do in holdTime
                disp('Rewarding well A')
                rewardWellAB = rewardPumpA
                trigger(2)
                lastWellAB = 1
                countAB = countAB + 1
                disp(countAB)
                trigger(6)
            end
        end
        if (lockout_enabled == 1) do in holdTime
            lockoutAB = 0
        end
    end
end;

callback portin[8] down
    disp('Unpoke wellA')
    currWellAB = 0
end;

callback portin[4] up % Well B
    disp('Poke in wellB')
    currWellAB = 2
    if (lockoutAB == 0) do
        lockoutAB = 1
        if (lastWellAB != 2) do
            active_barrierAB = barrierB
            trigger(5)
            if (currWellAB == 2) do in holdTime
                disp('Rewarding well B')
                rewardWellAB = rewardPumpB
                trigger(2)
                lastWellAB = 2
                countAB = countAB + 1
                disp(countAB)
                trigger(6)
            end
        end
        if (lockout_enabled == 1) do in holdTime
            lockoutAB = 0
        end
    end
end;

callback portin[4] down
    disp('Unpoke wellB')
    currWellAB = 0
end;
