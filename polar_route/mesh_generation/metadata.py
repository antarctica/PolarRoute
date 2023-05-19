




class Metadata:
    """
    A Metadata is a class that defines the datasource for a certain Cellbox and the assocated splitting conditions.


    Attributes:
       data_loader : object of the DataLoader class that enable projecting to the cellbox data
       splitting_conditions: list of conditions that determine how to split the data accessed by data_loader 
       value_fill_type (string): indicates how to fill a CellBox if it has void data
         (ex. use the data in the parent cellbox or assign 0 to the data)

    """
   

    def __init__(self, data_loader , spliiting_conditions =None, value_fill_type= ""):
        """

            Args:
               data_loader (DataLoader): object of the DataLoader class that enables projecting to the cellbox data
               splitting_conditions (List<dict>): list of conditions that determine how to split CellBox
               value_fill_tyep (string): represents the way the data of a cellbox will be filled
                 in case it has void data (ex. parent , 0 )
                
        """ 
        self.data_loader = data_loader
        self.splitting_conditions = spliiting_conditions
        self.value_fill_type = value_fill_type
  
    def get_data_loader(self): 
        """
        returns the data loader
        """
        return self.data_loader

    def get_splitting_conditions(self):
        """
        returns a list of the splitting conditions
        """
        
        return self.splitting_conditions

    def set_data_loader(self , data_loader): 
        """
        sets the data loader
        """
        self.data_loader = data_loader
 

    def set_splitting_conditions(self ,  splitting_conditions):
        """
        sets the splitting conditions
        """
        self.splitting_conditions = splitting_conditions

    def set_value_fill_type(self , value_fill_type): 
        """
        sets the value fill type
        """
        self.value_fill_type = value_fill_type

    def get_value_fill_type(self ):
        """
        returns thevalue fill type
        """
        return self.value_fill_type   
  

