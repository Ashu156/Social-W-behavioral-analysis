%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Protocol: W-Maze Spatial Alternation with Barrier Control
% Author: Edward L. Rivera, Ashutosh Shukla
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
int barrier_down_delay  = 20000    % Barrier remains up for this duration (ms)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Hardware Port Definitions – Barriers and Pumps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int rewardPump1 = 1      % Left well reward pump
int rewardPump2 = 5      % Center well reward pump
int rewardPump3 = 3      % Right well reward pump

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
    	disp('Left barrier UP')
    else if currWell == 2 do
        active_barrier = barrier2_center
	portout[active_barrier] = 1
    	disp('Center barrier UP')
    else if currWell == 3 do
        active_barrier = barrier3_right
	portout[active_barrier] = 1
    	disp('Right barrier UP')
    end
    
end;

function 5  % Lower barrier after delay
    do in barrier_down_delay
        portout[active_barrier] = 0
	if currWell == 1 do
        	disp('Left barrier DOWN')
	else if currWell == 2 do
		disp('Center barrier DOWN')
	else if currWell == 3 do
		disp('Right barrier DOWN')
	end
		
    end
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Callback Definitions – IR Sensor Ports
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

callback portin[10] up  % Left well
    disp('Poke in well 1 - LEFT')
    currWell = 1
    trigger(3)  % Handles first-time reward

    if lastWell == 0 do
        disp('Left well rewarded')
        rewardWell = rewardPump1
        trigger(1)
        count = count + 1
       disp(count)
        trigger(4)
        trigger(5)
    else if lastWell == 2 || lastSideWell == 3 do
        disp('Left well rewarded')
        rewardWell = rewardPump1
        trigger(1)
        count = count + 1
       disp(count)
        trigger(4)
        trigger(5)
    else do
        disp('Left well not rewarded')
    end
end;


callback portin[10] down
    disp('Unpoke in well 1 - LEFT')
    lastWell = 1
    lastSideWell = 1
end;

callback portin[14] up  % Center well
    disp('Poke in well 2 - CENTER')
    currWell = 2
    trigger(2)

    if lastWell == 0 do
        disp('Center well rewarded')
        rewardWell = rewardPump2
        trigger(1)
        count = count + 1
       disp(count)
        trigger(4)
        trigger(5)
    else if lastSideWell == 1 && (lastSideWell == 0 || lastSideWell == 3) do
        disp('Center well rewarded')
        rewardWell = rewardPump2
        trigger(1)
        count = count + 1
       disp(count)
        trigger(4)
        trigger(5)
    else do
        disp('Center well not rewarded')
    end
end;

callback portin[14] down
    disp('Unpoke in well 2 - CENTER')
    lastWell = 2
end;

callback portin[9] up  % Right well
    disp('Poke in well 3 - RIGHT')
    currWell = 3
    trigger(3)  % Handles first-time reward

    if lastWell == 0 do
        disp('Right well rewarded')
        rewardWell = rewardPump3
        trigger(1)
        count = count + 1
       disp(count)
        
        trigger(4)
        trigger(5)
       

    else if lastSideWell == 1 && (lastSideWell == 0 || lastWell == 2) do
        disp('Right well rewarded')
        rewardWell = rewardPump3
        trigger(1)
        count = count + 1
       disp(count)
        trigger(4)
        trigger(5)
    else do
        disp('Right well not rewarded')
    end
end;


callback portin[9] down
    disp('Unpoke in well 3 - RIGHT')
    lastWell = 3
    lastSideWell = 3
end;
