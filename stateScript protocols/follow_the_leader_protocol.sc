%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Timer Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Protocol: W-Maze Follow-the-Leader Task
% Author: Ashutosh Shukla
% Description:
%   - Extends the W-maze spatial alternation task for two animals
%   - One animal (leader) follows the original alternation protocol
%   - The second animal (follower) is rewarded for reaching same arm as leader
%   - Separate maze hardware and tracking for each animal
%   - Leader's choice is tracked and sets reward location for follower
%   - Barriers raise for both correct and incorrect choices
%
% Key Parameters:
%   deliverPeriod: Reward duration (ms)
%   barrier_down_delay: Time (ms) barrier remains up after reward or incorrect choice
%   follower_window: Time window (ms) for follower to make matching choice
%
% Output Ports:
%   Leader Reward Pumps: leaderPump1 (left), leaderPump2 (center), leaderPump3 (right)
%   Follower Reward Pumps: followerPump1 (left), followerPump2 (center), followerPump3 (right)
%   Leader Barriers: leaderBarrier1 (left), leaderBarrier2 (center), leaderBarrier3 (right)
%   Follower Barriers: followerBarrier1 (left), followerBarrier2 (center), followerBarrier3 (right)
%
% Input Ports:
%   Leader IR sensors: leaderSensor1 (left), leaderSensor2 (center), leaderSensor3 (right)
%   Follower IR sensors: followerSensor1 (left), followerSensor2 (center), followerSensor3 (right)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adjustable Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int deliverPeriod = 600     % Reward duration (ms)
int holdTime = 100          % Hold time requirement (not used currently)
int barrier_down_delay = 20000  % Barrier remains up for this duration (ms)
int follower_window = 30000     % Time window for follower to make choice (ms)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Hardware Port Definitions - Leader Animal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int leaderPump1 = 1       % Left well reward pump for leader
int leaderPump2 = 5       % Center well reward pump for leader
int leaderPump3 = 3       % Right well reward pump for leader

int leaderBarrier1 = 18   % Left arm barrier port for leader
int leaderBarrier2 = 17   % Center arm barrier port for leader
int leaderBarrier3 = 19   % Right arm barrier port for leader

int leaderActiveBarrier = 0   % Tracks which barrier is currently active for leader

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Hardware Port Definitions - Follower Animal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int followerPump1 = 2     % Left well reward pump for follower
int followerPump2 = 6     % Center well reward pump for follower
int followerPump3 = 4     % Right well reward pump for follower

int followerBarrier1 = 21   % Left arm barrier port for follower
int followerBarrier2 = 20   % Center arm barrier port for follower
int followerBarrier3 = 22   % Right arm barrier port for follower


int followerActiveBarrier = 0   % Tracks which barrier is currently active for follower

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Leader Behavioral Tracking Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int leaderLastSideWell = 0  % 1 if left, 3 if right - tracks last visited side well
int leaderLastWell = 0      % 1 = left, 2 = center, 3 = right - last visited well
int leaderCurrWell = 0      % Current active well during a poke

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Follower Behavioral Tracking Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int followerLastWell = 0    % 1 = left, 2 = center, 3 = right - last visited well
int followerCurrWell = 0    % Current active well during a poke
int targetWell = 0          % Well that follower should aim for (set by leader)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Task State Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int leaderRewardWell = 0    % Currently active reward pump for leader
int followerRewardWell = 0  % Currently active reward pump for follower
int leaderNowRewarding = 0  % Flag for leader reward delivery state
int followerNowRewarding = 0  % Flag for follower reward delivery state
int leaderRewardCount = 0   % Total reward count for leader
int followerRewardCount = 0 % Total reward count for follower
int choiceWindowActive = 0  % Indicates if follower choice window is active

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reward and Barrier Control Functions - Leader
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function 1  % Deliver reward to leader
    leaderNowRewarding = 1
    portout[leaderRewardWell] = 1
    do in deliverPeriod
        portout[leaderRewardWell] = 0
        leaderNowRewarding = 0
    end
end;

function 2  % Reward leader center on first visit
    if leaderLastWell == 0 do
        leaderRewardWell = leaderCurrWell
        trigger(1)
    end
end;

function 3  % Reward leader left or right on first visit
    if leaderLastSideWell == 0 && (leaderCurrWell == 1 || leaderCurrWell == 3) do
        leaderRewardWell = leaderCurrWell
        trigger(1)
    end
end;

function 4  % Raise correct barrier for leader based on leaderCurrWell
    if leaderCurrWell == 1 do
        leaderActiveBarrier = leaderBarrier1
        portout[leaderActiveBarrier] = 1
        disp('Leader left barrier UP')
    else if leaderCurrWell == 2 do
        leaderActiveBarrier = leaderBarrier2
        portout[leaderActiveBarrier] = 1
        disp('Leader center barrier UP')
    else if leaderCurrWell == 3 do
        leaderActiveBarrier = leaderBarrier3
        portout[leaderActiveBarrier] = 1
        disp('Leader right barrier UP')
    end
end;

function 5  % Lower leader barrier after delay
    do in barrier_down_delay
        portout[leaderActiveBarrier] = 0
        if leaderCurrWell == 1 do
            disp('Leader left barrier DOWN')
        else if leaderCurrWell == 2 do
            disp('Leader center barrier DOWN')
        else if leaderCurrWell == 3 do
            disp('Leader right barrier DOWN')
        end
    end
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reward and Barrier Control Functions - Follower
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function 6  % Deliver reward to follower
    followerNowRewarding = 1
    portout[followerRewardWell] = 1
    do in deliverPeriod
        portout[followerRewardWell] = 0
        followerNowRewarding = 0
    end
end;

function 7  % Raise correct barrier for follower based on followerCurrWell
    if followerCurrWell == 1 do
        followerActiveBarrier = followerBarrier1
        portout[followerActiveBarrier] = 1
        disp('Follower left barrier UP')
    else if followerCurrWell == 2 do
        followerActiveBarrier = followerBarrier2
        portout[followerActiveBarrier] = 1
        disp('Follower center barrier UP')
    else if followerCurrWell == 3 do
        followerActiveBarrier = followerBarrier3
        portout[followerActiveBarrier] = 1
        disp('Follower right barrier UP')
    end
end;

function 8  % Lower follower barrier after delay
    do in barrier_down_delay
        portout[followerActiveBarrier] = 0
        if followerCurrWell == 1 do
            disp('Follower left barrier DOWN')
        else if followerCurrWell == 2 do
            disp('Follower center barrier DOWN')
        else if followerCurrWell == 3 do
            disp('Follower right barrier DOWN')
        end
    end
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional Global Variables for Time Window
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int choiceWindowStartTime = 0  % Timestamp when choice window starts
int currentTime = 0            % For storing current time when needed
int timeDiff = 0               % For calculating time differences

function 9  % Set target well for follower
     targetWell = leaderCurrWell    % Set target based on leader's current well
     choiceWindowActive = 1

     if choiceWindowActive == 1 do in follower_window
         choiceWindowActive = 0
    end
     

     % Set follower's reward well based on target
    if targetWell == 1 do
        followerRewardWell = followerPump1
    else if targetWell == 2 do
        followerRewardWell = followerPump2
     else if targetWell == 3 do
         followerRewardWell = followerPump3
end
end;

function 10  % Check if follower made correct choice
   
   % Now handle follower's choice
   if choiceWindowActive == 1 do
       if followerCurrWell == targetWell do
           % Reward follower for correct choice
           trigger(6)
           followerRewardCount = followerRewardCount + 1
          disp(followerRewardCount)
        
           % Mark choice window as complete
           choiceWindowActive = 0
        
           % Raise and later lower barrier
           trigger(7)
           trigger(8)
       else if (followerLastWell != followerCurrWell) && (followerCurrWell!=targetWell) do
           disp('Follower made INCORRECT choice')
           % Incorrect choice - no reward but still raise barrier
           trigger(7)
           trigger(8)
       end
    end
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Callback Definitions - Leader IR Sensors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

callback portin[10] up  % Leader left well
    disp('Leader poke in well 1 - LEFT')
    leaderCurrWell = 1
    trigger(3)  % Handles first-time reward

    if leaderLastWell == 0 do
        %disp('Leader left well rewarded')
        leaderRewardWell = leaderPump1
        trigger(1)
        leaderRewardCount = leaderRewardCount + 1
        disp(leaderRewardCount)
        trigger(4)
        trigger(5)
        
        % Set target for follower
        trigger(9)
    else if leaderLastWell == 2 && (leaderLastSideWell == 0 || leaderLastSideWell == 3) do
        %disp('Leader left well rewarded')
        leaderRewardWell = leaderPump1
        trigger(1)
        leaderRewardCount = leaderRewardCount + 1
        disp(leaderRewardCount)
        trigger(4)
        trigger(5)
        
        % Set target for follower
        trigger(9)
    else if leaderLastWell != 1 do
        disp('Leader left well not rewarded - incorrect choice')
        trigger(4)  % Raise barrier
        trigger(5)  % Lower after delay
    else do
        %disp('Leader left well not rewarded - repeated poke')
    end
end;

callback portin[10] down
    disp('Leader unpoke in well 1 - LEFT')
    leaderLastWell = 1
    leaderLastSideWell = 1
end;

callback portin[14] up  % Leader center well
    disp('Leader poke in well 2 - CENTER')
    leaderCurrWell = 2
    trigger(2)

    if leaderLastWell == 0 do
        %disp('Leader center well rewarded')
        leaderRewardWell = leaderPump2
        trigger(1)
        leaderRewardCount = leaderRewardCount + 1
        disp(leaderRewardCount)
        trigger(4)
        trigger(5)
        
        % Set target for follower
        trigger(9)
    else if leaderLastWell == 1 || leaderLastWell == 3 do
        %disp('Leader center well rewarded')
        leaderRewardWell = leaderPump2
        trigger(1)
        leaderRewardCount = leaderRewardCount + 1
        disp('Leader reward count:')
        disp(leaderRewardCount)
        trigger(4)
        trigger(5)
        
        % Set target for follower
        trigger(9)
    else if leaderLastWell != 2 do
        disp('Leader center well not rewarded - incorrect choice')
        trigger(4)  % Raise barrier
        trigger(5)  % Lower after delay
    else do
        %disp('Leader center well not rewarded - repeated poke')
    end
end;

callback portin[14] down
    disp('Leader unpoke in well 2 - CENTER')
    leaderLastWell = 2
end;

callback portin[9] up  % Leader right well
    disp('Leader poke in well 3 - RIGHT')
    leaderCurrWell = 3
    trigger(3)  % Handles first-time reward

    if leaderLastWell == 0 do
        %disp('Leader right well rewarded')
        leaderRewardWell = leaderPump3
        trigger(1)
        leaderRewardCount = leaderRewardCount + 1
        disp('Leader reward count:')
        disp(leaderRewardCount)
        trigger(4)
        trigger(5)
        
        % Set target for follower
        trigger(9)
    else if leaderLastWell == 2 && (leaderLastSideWell == 0 || leaderLastSideWell == 1) do
        %disp('Leader right well rewarded')
        leaderRewardWell = leaderPump3
        trigger(1)
        leaderRewardCount = leaderRewardCount + 1
        disp('Leader reward count:')
        disp(leaderRewardCount)
        trigger(4)
        trigger(5)
        
        % Set target for follower
        trigger(9)
    else if leaderLastWell != 3 do
        disp('Leader right well not rewarded - incorrect choice')
        trigger(4)  % Raise barrier
        trigger(5)  % Lower after delay
    else do
        %disp('Leader right well not rewarded - repeated poke')
    end
end;

callback portin[9] down
    disp('Leader unpoke in well 3 - RIGHT')
    leaderLastWell = 3
    leaderLastSideWell = 3
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Callback Definitions - Follower IR Sensors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

callback portin[13] up  % Follower left well
    disp('Follower poke in well 1 - LEFT')
    followerCurrWell = 1
    if (followerLastWell!=1) do
        trigger(10)
    end
end;

callback portin[13] down
    disp('Follower unpoke in well 1 - LEFT')
    followerLastWell = 1
end;

callback portin[4] up  % Follower center well
    disp('Follower poke in well 2 - CENTER')
    followerCurrWell = 2
    if (followerLastWell!=2) do
        trigger(10)
    end
end;

callback portin[4] down
    disp('Follower unpoke in well 2 - CENTER')
    followerLastWell = 2
end;

callback portin[8] up  % Follower right well
    disp('Follower poke in well 3 - RIGHT')
    followerCurrWell = 3
    if (followerLastWell!=3) do
        trigger(10)
    end
end;

callback portin[8] down
    disp('Follower unpoke in well 3 - RIGHT')
    followerLastWell = 3
end;