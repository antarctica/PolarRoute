
class SpeedFunctions:
    def __init_(self,config):
        '''
            FILL
        '''


    def ice_resistance(self,config):
        """
                Function to find the ice resistance force at a given speed in a given cell.

                Inputs:
                cell - Cell box object

                Outputs:
                resistance - Resistance force
        """
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

        hull = self.config['Vehicle_Info']['HullType']
        beam = self.config['Vehicle_Info']['Beam']
        kparam, bparam, nparam = hull_params[hull]
        gravity = 9.81  # m/s-2
        speed = self.config['Vehicle_Info']['Speed']*(5./18.)  # assume km/h and convert to m/s
        force_limit = speed/np.sqrt(gravity*cell.iceArea()*cell.iceThickness(self.config['Region']['startTime']))
        resistance = 0.5*kparam*(force_limit**bparam)*cell.iceDensity(self.config['Region']['startTime'])*beam*cell.iceThickness(self.config['Region']['startTime'])*(speed**2)*(cell.iceArea()**nparam)
        return resistance

    def inverse_resistance(self, force_limit, cell):
        """
        Function to find the speed that keeps the ice resistance force below a given threshold.

        Inputs:
        force_limit - Force limit
        cell        - Cell box object

        Outputs:
        speed - Vehicle Speed
        """
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

        hull = self.config['Vehicle_Info']['HullType']
        beam = self.config['Vehicle_Info']['Beam']
        kparam, bparam, nparam = hull_params[hull]
        gravity = 9.81  # m/s-2

        vexp = 2*force_limit/(kparam*cell.iceDensity(self.config['Region']['startTime'])*beam*cell.iceThickness(self.config['Region']['startTime'])*(cell.iceArea()**nparam)*(gravity*cell.iceThickness(self.config['Region']['startTime'])*cell.iceArea())**-(bparam/2))

        vms = vexp**(1/(2.0 + bparam))
        speed = vms*(18./5.)  # convert from m/s to km/h

        return speed

    def speed(self, cell):
        '''
            FILL
        '''
        if self.variable_speed:
            if cell.iceArea() == 0.0:
                speed = self.config['Vehicle_Info']['Speed']
            elif self.ice_resistance(cell) < self.config['Vehicle_Info']['ForceLimit']:
                speed = self.config['Vehicle_Info']['Speed']
            else:
                speed = self.inverse_resistance(self.config['Vehicle_Info']['ForceLimit'], cell)
        else:
            speed = self.config['Vehicle_Info']['Speed']
        return speed
