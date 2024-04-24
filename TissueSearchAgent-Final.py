# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:00:03 2024

@author: tomas
"""

if __name__ == "__main__":
    
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms
    import torchvision
    import torch.distributions as torchdist
    import torch.nn as nn
    import time
    from PIL import Image, ImageOps, ImageFilter
    import random
    import torch.optim as optim
    from tqdm import tqdm
    import pandas as pd
    import os
    import openslide
    import cv2
    import sys
    from tiatoolbox.wsicore.wsireader import WSIReader
    
    # Command-line Arguments
    wsiInputPath1 = sys.argv[1]
    wsiInputPath2 = sys.argv[2]
    outputPath = sys.argv[3]
    workspacePath = sys.argv[4]
    #-------------------------------
    

    def stitchSlide(pathToPatches, patchSize):
        fileName = pathToPatches.split("\\")[-1]
        slide = fileName.split("/")[-1]
        patch_files = [os.path.join(pathToPatches, file) for file in (os.listdir(pathToPatches)[3:]) if file.endswith('.jpg')]
    
        # Create a dict mapping each x,y coordinate to the given patches path
        # Additionally stores the maximum x and y coordinates to know the final size of the image
        max_x = 0
        max_y = 0
        for patch in patch_files:
            patchName = os.path.basename(patch).split(".")[0]
            if(patchName == "slide_thumbnail"):
              continue
            x, y = map(int, patchName.split("_")[-2:])
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    
        stitched_img = np.empty((max_y + patchSize, max_x + patchSize, 3), dtype=np.uint8)
    
        # Creating the full array representing the original image
        for patch in patch_files:
            patchName = os.path.basename(patch).split(".")[0]
            if(patchName == "slide_thumbnail"):
              continue
            x, y = map(int, patchName.split("_")[-2:])
            with Image.open(patch) as img:
                patch_array = np.array(img)
    
                # Check the shape and pad if necessary
                if patch_array.shape != (patchSize, patchSize, 3):
                    padded_patch = np.zeros((patchSize, patchSize, 3), dtype=np.uint8)
                    padded_patch[:patch_array.shape[0], :patch_array.shape[1]] = patch_array
                    patch_array = padded_patch
    
                stitched_img[y:y + patchSize, x:x + patchSize] = patch_array
    
        return stitched_img
    
    

    class SlideGridEnv2():

      def __init__(self, slideList, patchSize, magnification, workspacePath):
        self.slideList = slideList
        self.currentSlide = 0
        self.patch_size = patchSize
        self.magnification_level = magnification
        self.workspacePath = workspacePath
        self.currentGrid = 0
        self.currentState = (0,0)
        self.prevState = None
        self.currentBaseline = 0
        self.visitedStates = []



      def getGrid(self):
        return self.currentGrid

      def reset(self):

        # Switch WSI
        try:
          currentSlidePath = str(self.slideList[self.currentSlide])
        except IndexError:
          state = None
          episodeEnd = True
          print("-----Episode Ended IDX -----")
          return state, episodeEnd
        self.currentState = (0,0)
        self.prevState = (0,0)
        self.visitedStates = []
        episodeEnd = False

        # Generate Current Grid
        wsiReader = WSIReader.open(input_img=currentSlidePath)
        try:
          wsiReader.save_tiles(output_dir=self.workspacePath, tile_objective_value=self.magnification_level, tile_read_size=(512,512))
        except FileExistsError:
          pass
        self.currentGrid = stitchSlide(str(self.workspacePath + "/" + self.slideList[self.currentSlide].split("/")[-1]), self.patch_size[0])
        self.terminalState = (int(self.currentGrid.shape[0]/self.patch_size[0]), int(self.currentGrid.shape[1]/self.patch_size[1]))
        print("TSTATE", self.terminalState)

        # Get initial state
        state = self.currentGrid[self.currentState[0]:self.patch_size[0],self.currentState[1]:self.patch_size[1]]

        # Generate the baseline for comparison
        self.currentBaseline = np.mean(state[:,:,:], axis=(0,1))

        # Prepare the Next Slide
        self.currentSlide = self.currentSlide + 1

        print("NEW SLIDE - SHAPE:", self.currentGrid.shape)

        # Adding the grid info
        gridVisual = torch.zeros([self.terminalState[0], self.terminalState[1]])
        gridVisual[self.currentState[0], self.currentState[1]] = 1
        gridVisual_resized = nn.functional.interpolate(gridVisual.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
        gridVisual_resized = torch.transpose(gridVisual_resized, 0,2)

        firstState = torch.cat((torch.tensor(state), torch.tensor(state), gridVisual_resized), 2)

        return firstState, episodeEnd

      def generateReward(self, state, done):
        # If Out-of-Bounds
        if done == True:
          reward = -100
          done = True
          return reward, done

        stateRGB = np.mean(state[:,:,:], axis=(0,1))
        if stateRGB[0]<self.currentBaseline[0]-3 and stateRGB[1]<self.currentBaseline[1]-3 and stateRGB[2]<self.currentBaseline[2]-3:
            reward = 1
        else:
            reward = -1

        done = False
        return reward, done

      def moveSpace(self,moveTuple):
        # Track Visited
        self.visitedStates.append(self.currentState)
        self.prevState = self.currentState
        print("Current", self.currentState, "MOVETUPLE", moveTuple)
        self.currentState = (self.currentState[0]+moveTuple[0], self.currentState[1]+moveTuple[1])
        print("NEW CURRENT", self.currentState)
        return self.currentState

      def step(self, action):
        print("BEFORE STEP VISITED", self.visitedStates)

        # Save the current state to see if we moved out of bounds
        self.prevState = self.currentState
        done = False

        # Remove Action from list in the case where it was an e-greedy random action
        if (type(action) == list):
          action = action[0]

        # Move to the next patch
        if action == 0:         # Up
          if self.currentState == self.terminalState or self.currentState[0] <= 0:
            newPos = self.currentState
            done = True
          else:
            newPos = self.moveSpace((-1, 0))
        elif action == 1:       # Down
          if self.currentState == self.terminalState or self.currentState[0] >= self.terminalState[0]:
            newPos = self.currentState
            done = True
          else:
            newPos = self.moveSpace((1, 0))
        elif action == 2:       # Left
          if self.currentState == self.terminalState or self.currentState[1] <= 0:
            newPos = self.currentState
            done = True
          else:
            newPos = self.moveSpace((0, -1))
        elif action == 3:       # Right
          if self.currentState == self.terminalState or self.currentState[1] >= self.terminalState[1]:
            newPos = self.currentState
            done = True
          else:
            newPos = self.moveSpace((0, 1))
        elif action == 4:       # Up-Left
          if self.currentState == self.terminalState or self.currentState[0] <= 0 or self.currentState[1] <= 0:
            newPos = self.currentState
            done = True
          else:
            newPos = self.moveSpace((-1, -1))
        elif action == 5:       # Up-Right
          if self.currentState == self.terminalState or self.currentState[0] <= 0 or self.currentState[1] >= self.terminalState[1]:
            newPos = self.currentState
            done = True
          else:
            newPos = self.moveSpace((-1, 1))
        elif action == 6:       # Down-Left
          if self.currentState == self.terminalState or self.currentState[0] >= self.terminalState[0] or self.currentState[1] <= 0:
            newPos = self.currentState
            done = True
          else:
            newPos = self.moveSpace((1, -1))
        elif action == 7:       # Down-Right
          if self.currentState == self.terminalState or self.currentState[0] >= self.terminalState[0] or self.currentState[1] >= self.terminalState[1]:
            newPos = self.currentState
            done = True
          else:
            newPos = self.moveSpace((1, 1))

        # After moving, get the state
        print("info", self.currentState[0]*self.patch_size[0], self.currentState[0]*self.patch_size[0]+self.patch_size[0], self.currentState[1]*self.patch_size[1], self.currentState[1]*self.patch_size[1]+self.patch_size[1],)
        state = self.currentGrid[self.currentState[0]:self.currentState[0]+self.patch_size[0],self.currentState[1]:self.currentState[1]+self.patch_size[1]]
        if self.prevState is None:
          prevState = self.currentGrid[self.currentState[0]:abs(self.currentState[0]+self.patch_size[0]),self.currentState[1]:abs(self.currentState[1]+self.patch_size[1])]
        else:
          prevState = self.currentGrid[self.prevState[0]:self.prevState[0]+self.patch_size[0],self.prevState[1]:self.prevState[1]+self.patch_size[1]]

        # Adding the grid info
        gridVisual = torch.zeros([self.terminalState[0], self.terminalState[1]])
        gridVisual[self.currentState[0], self.currentState[1]] = 1
        gridVisual_resized = nn.functional.interpolate(gridVisual.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
        gridVisual_resized = torch.transpose(gridVisual_resized, 0,2)

        # Contatenating to pass the current state and previous one
        stateTotal = torch.cat((torch.tensor(state), torch.tensor(prevState), gridVisual_resized), 2)

      ## Get reward
        # Hit the edge
        if done == True:
          reward = -100
        if self.currentState == self.terminalState:
          done = True
          reward = 0
        elif (self.currentState == (0,0) and self.currentState in self.visitedStates):
          done = True
          reward = -200
        elif self.currentState in self.visitedStates:
          reward = -10
          done = False
        else:
          reward, done = self.generateReward(state, done)


        print("stateTotal", stateTotal.shape)
        return stateTotal, reward, done
    
    
    class TissueSearchAgent2():
        def __init__(
            self,
            discount=0.99,
            epsilon=0.6,
            name="Actor-Critic",
        ):
    
          resnet_backbone = torchvision.models.resnet18(pretrained=True)
          self.replace_batchnorm_with_identity(resnet_backbone)
          # for layer in list(resnet_backbone.children()):
          #   print(layer)
          modulesPolicy = list(resnet_backbone.children())[:-1]
          modulesPolicy[0] = nn.Conv2d(7,64,(7,7),(2,2),(3,3),bias=False)
          modulesValue = list(resnet_backbone.children())[:-1]
          modulesValue[0] = nn.Conv2d(7,64,(7,7),(2,2),(3,3),bias=False)

          self.resnet_backboneP = nn.Sequential(*modulesPolicy)
          self.output_layerP = nn.Sequential(nn.Linear(512, 8), nn.Softmax())
          self.resnet_backboneV = nn.Sequential(*modulesValue)
          self.output_layerV = nn.Linear(1,1)

          self.policy_network = torch.nn.Sequential(self.resnet_backboneP, self.output_layerP)
          self.opt = torch.optim.Adam(self.policy_network.parameters(), lr=0.0004)
          self.value_network = torch.nn.Sequential(self.resnet_backboneV, self.output_layerV)
          self.value_opt = torch.optim.Adam(self.value_network.parameters(), lr=0.0004)

          self.discount = discount
          self.epsilon = epsilon
        
        # Because we are only passing 1 patch at a time to the models,
        # we do not have to have any batchnorm effect
        # This can cause instability in the model.
        # We opt to replace the BN layers from the pretrained model with Identity.
        def replace_batchnorm_with_identity(self, model):
          for child_name, child in model.named_children():
              if isinstance(child, torch.nn.BatchNorm2d):
                  # Replace the BatchNorm layer with Identity
                  setattr(model, child_name, torch.nn.Identity())
              else:
                  # Otherwise, proceed to replace in child modules
                  self.replace_batchnorm_with_identity(child)
    
        def forward(self, state):
          x = state.clone().unsqueeze(0)
          x = torch.transpose(x, 1, 3)
          x = self.resnet_backboneP(x)
          y = x.view(x.size(0), -1)  # Flatten the output
          z = self.output_layerP(y)
          return z
    
        def distribution(self, state):
          probs = self.forward(state)
          dist = torchdist.categorical.Categorical(probs=probs)
          return dist
    
        def action(self, state):
    
          if random.random() < self.epsilon:
            action = random.sample([0,1,2,3,4,5,6,7],1)
            print(action)
            return action
    
          dist = self.distribution(state)
          action = dist.sample()
          print(action)
          action = action.item()
          return action
    
        def train_episode(self, env) -> float:
            loss_dict = {}
    
            episodeEnd = False
            while not episodeEnd:
              done = False
              loss_dict["policy_loss"] = 0
              loss_dict["value_loss"] = 0
              cumRewards = []
              t = 0
              I = 1
              state, episodeEnd = env.reset()
              if (episodeEnd):
                break
              cumulativeReward = 0
              while not done:
                print("NEW STEP --")
                state = torch.tensor(state, dtype=torch.float32)
                # Get action
                action = self.action(state)
                # Get state value
                valueState = torch.transpose(state.unsqueeze(0), 1,3)
                stateValue = self.value_network(valueState).detach()
                # Take step
                stateNextPre, reward, done = env.step(action)
                stateNext = torch.tensor(stateNextPre, dtype=torch.float32)
                cumulativeReward += reward
                if t >= 20:
                  done = True
                print("REWARD", reward)
                # Next State Value
                if not done:
                  inputState = torch.transpose(stateNext.unsqueeze(0), 1,3)
                  stateValueNext = self.value_network(inputState).detach() # Detach for stop_gradient
                else:
                  stateValueNext = torch.zeros_like(stateValue)
                  print("CUM REWARD", cumulativeReward)
                # State Action Prob
                prob = self.distribution(state).log_prob(torch.tensor(action, dtype=torch.float32))
                # Losses
                policyInner = reward + (self.discount*stateValueNext.detach()) - stateValue
                policyLoss = -policyInner*prob*I
                valueLoss = torch.nn.functional.mse_loss(stateValue, (reward + (self.discount*stateValueNext.detach())))/2
                # Updates
                self.opt.zero_grad()
                pLossSum = policyLoss.sum()
                pLossSum.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
                self.opt.step()
                self.value_opt.zero_grad()
                valueForGrad = torch.tensor([valueLoss.item()], requires_grad=True)
                valueForGrad.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
                self.value_opt.step()
                # Track
                loss_dict["policy_loss"] += pLossSum.item()
                loss_dict["value_loss"] += valueForGrad.item()
                # Loop Updates
                state = stateNext
                state = np.array(state)
                t += 1
                I = I*self.discount
    
    
              loss_dict["policy_loss"] /= t
              loss_dict["value_loss"] /= t
              print("PD",loss_dict["policy_loss"])
              print("LD",loss_dict["value_loss"])
              loss_dict["reward"] = cumulativeReward
    
            return loss_dict
      
        
      
    # EXECUTION
        
    path1 = wsiInputPath1
    path2 = wsiInputPath2
    
    allSlides = []
    path1List = os.listdir(wsiInputPath1)
    path2List = os.listdir(wsiInputPath2)
    
    for path in path1List:
      allSlides.append(str(path1 + "/" + path))
    
    for path in path2List:
      allSlides.append(str(path2 + "/" + path))      
      
      
    rewards = []
    myEnv = SlideGridEnv2(allSlides, patchSize=(512,512,), magnification=5, workspacePath=workspacePath)
    myAgent = TissueSearchAgent2()
    numEpochs = 25
    for epoch in range(numEpochs):
      loss_dict = myAgent.train_episode(myEnv)
      rewards.append(loss_dict["reward"])
      myEnv = SlideGridEnv2(allSlides, patchSize=(512,512,), magnification=5, workspacePath=workspacePath)
      
      
      
    plt.plot(np.arange(0,numEpochs), rewards)
    plt.title("Cumulative Reward Per Episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Episodic Reward")
    plt.savefig(str(outputPath + "/CumReward"))   #---------

    torch.save(myAgent.policy_network.state_dict(), str(outputPath + "/pnet.pth"))
    torch.save(myAgent.value_network.state_dict(), str(outputPath + "/vnet.pth"))
    
    
    
    