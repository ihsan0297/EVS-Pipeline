class Student:
    def __init__(self, name, marks):
        self.name=name
        self.marks=marks

    def get_avg(self):
        sum=0
        for val in self.marks:
             sum += val

             print("hi", self.name, "your average marks is:", sum/3)
             




        student1= Student("ali", [99, 100, 97, 96, 93])
        student1.get_vag()