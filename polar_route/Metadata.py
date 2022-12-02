




class Metadata:
    """
    A Metadata is a class that defines the datasource for a certain Cellbox and the assocated splitting conditions.


    Attributes:
       data_loader : object of the DataLoader class that enable projecting to the cellBox data
       splitting_conditions: list of conditions that determine how to split the data accessed by data_loader 
       aggregate_type (string): the type of aggrgation applied to CellBox data (ex. Min, Max ,..)

    """
   

    def __init__(self, data_loader , spliiting_conditions, aggregate_type):
        """

            Args:
               data_loader (DataLoader): object of the DataLoader class that enables projecting to the cellBox data
               splitting_conditions (List<dict>): list of conditions that determine how to split CellBox
               aggregate_type (string): the type of aggrgation applied to CellBox data (ex. Min, Max ,..)
                
        """
        # Boundary information 
        self.data_loader = data_loader
        self.splitting_conditions = spliiting_conditions
        self.aggregate_type = aggregate_type
       


  
    def get_data_loader(self): 
        return self.data_loader

    def get_splitting_conditions(self):
        
        return self.splitting_conditions

    def set_data_loader(self , data_loader): 
        self.data_loader = data_loader

    def get_aggregate_type(self): 
        return self.aggregate_type


    def set_aggregate_type(self , agg_type): 
        self.aggregate_type = agg_type  

    def set_splitting_conditions(self ,  splitting_conditions):
        self.splitting_conditions = splitting_conditions
  

