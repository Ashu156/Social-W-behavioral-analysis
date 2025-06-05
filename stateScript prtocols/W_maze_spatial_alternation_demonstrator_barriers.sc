%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Protocol: W-Maze Spatial Alternation with Barrier Control
% Author: Edward L. Rivera
% Description:
%   - Implements a W-maze task with reward-based alternation.
%   - Motorized barriers remain down until the animal makes a correct choice.
%   - Upon correct choice and reward, the corresponding barrier lifts.
%   - Barriers stay up for a configurable delay (barrier_down_delay).
%
% Key Parameters:
%   deliverPeriod: Reward duration (ms)
%   barrier_down_delay: Time (ms) barrier remains up after reward
%
% Output Ports:
%   Reward Pumps: rewardPump1 (port 1), rewardPump2 (port 2), rewardPump3 (port 3)
%   Barriers: barrier1 (port 6), barrier2 (port 7), barrier3 (port 8)
%
% Notes:
%   Use 'active_barrier' to dynamically assign which barrier to control.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adjustable Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int deliverPeriod = 600     % Reward duration (ms)
int holdTime = 100    % Hold time requirement (not used currently)
int barrier_down_delay  = 2000    % Barrier remains up for this duration (ms)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Hardware Port Definitions – Barriers and Pumps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int rewardPump1 = 1      % Left well reward pump
int rewardPump2 = 3      % Center well reward pump
int rewardPump3 = 5      % Right well reward pump

int barrier1_left   = 18  % Left arm barrier port
int barrier2_center = 17  % Center arm barrier port
int barrier3_right  = 19  % Right arm barrier port

int active_barrier = 0   % Tracks which barrier is currently active

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Behavioral Tracking Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int lastSideWell = 0     % 1 if left, 3 if right – tracks last visited side well
int lastWell = 0            %1 = left, 2 = center, 3 = right – last visited well
int currWell = 0           % Current active well during a poke

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Task State Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int rewardWell    = 0    % Currently active reward pump
int nowRewarding  = 0    % Flag for reward delivery state
int count         = 0    % Total reward count

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reward and Barrier Control Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function 1  % Deliver reward
    nowRewarding = 1
    portout[rewardWell] = 1
    do in deliverPeriod
        portout[rewardWell] = 0
        nowRewarding = 0
    end
end;

function 2  % Reward center on first visit
    if lastWell == 0 do
        rewardWell = currWell
        trigger(1)
    end
end;

function 3  % Reward left or right on first visit
    if lastSideWell == 0 && (currWell == 1 || currWell == 3) do
        rewardWell = currWell
        trigger(1)
    end
end;

function 4  % Raise correct barrier based on currWell
    if currWell == 1 do
        active_barrier = barrier1_left
        portout[active_barrier] = 1
    	disp('Barrier 1-Left UP')
    else if currWell == 2 do
        active_barrier = barrier2_center
	portout[active_barrier] = 1
    	disp('Barrier 2-Center UP')
    else if currWell == 3 do
        active_barrier = barrier3_right
	portout[active_barrier] = 1
    	disp('Barrier 3-Right UP')
    end
    
end;

function 5  % Lower barrier after delay
    do in barrier_down_delay
        portout[active_barrier] = 0
	if currWell == 1 do
        	disp('Barrier 1-Left DOWN')
	else if currWell == 2 do
		disp('Barrier 2-Center DOWN')
	else if currWell == 3 do
		disp('Barrier 3-Right DOWN')
	end
		
    end
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Callback Definitions – IR Sensor Ports
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

callback portin[10] up  % Left well
    disp('Portin1 up - Left well on')
    currWell = 1
    trigger(3)

    if lastWell == 2 do
        if lastSideWell == 3 do
            disp('Poke 1 rewarded - left')
            rewardWell = rewardPump1
            trigger(1)
            trigger(4)
            trigger(5)
        end
    else do
        disp('Poke 1 not rewarded - left')
    end
end;

callback portin[10] down
    disp('Portin1 down - Left well off')
    lastWell = 1
    lastSideWell = 1
end;

callback portin[15] up  % Center well
    disp('Portin2 up - Center well on')
    currWell = 2
    trigger(2)

    if lastWell == 1 || lastWell == 3 do
        disp('Poke 2 rewarded - center')
        rewardWell = rewardPump2
        trigger(1)
        trigger(4)
        trigger(5)
    else do
        disp('Poke 2 not rewarded - center')
    end
end;

callback portin[15] down
    disp('Portin2 down - Center well off')
    lastWell = 2
end;

callback portin[1] up  % Right well
    disp('Portin3 up - Right well on')
    currWell = 3
    trigger(3)

    if lastWell == 2 do
        if lastSideWell == 1 do
            disp('Poke 3 rewarded - right')
            rewardWell = rewardPump3
            trigger(1)
            trigger(4)
            trigger(5)
        else do
            disp('Poke 3 not rewarded - right')
        end
    end
end;

callback portin[1] down
    disp('Portin3 down - Right well off')
    lastWell = 3
    lastSideWell = 3
end;
