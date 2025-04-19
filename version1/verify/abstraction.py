"""
An abstract class can be considered as a blueprint for other classes. 
It allows you to create a set of methods that must be created within any child classes built from the abstract class. 
A class which contains one or more abstract methods is called an abstract class. 
An abstract method is a method that has a declaration but does not have an implementation.
 While we are designing large functional units we use an abstract class. 
 When we want to provide a common interface for different implementations of a component, we use an abstract class. 
  
Why use Abstract Base Classes : 
By defining an abstract base class, you can define a common Application Program Interface(API) for a set of subclasses. 
This capability is especially useful in situations where a third-party is going to provide implementations, such as with plugins, 
but can also help you when working in a large team or with a large code-base where keeping all classes in your mind is difficult or not possible. 
"""



from abc import ABC , abstractmethod

class Polygon(ABC):

    @abstractmethod
    def noofsides(self):
        pass

class triangle(Polygon):

    def noofsides(self):
        print("I have 3 sides")


class quadrilateral(Polygon):

    def noofsides(self):
        print("I have 4 sides")

class hexagonal(Polygon):

    def noofsides(self):
        print("I have 6 sides")


trian = triangle()
trian.noofsides()

quad = quadrilateral()
quad.noofsides()

hex = hexagonal()
hex.noofsides()














# Example 2


class Animal(ABC):

    @abstractmethod
    def move(self):
        pass


class Lion(Animal):
    def move(self):
        print("I can run")


class Eagle(Animal):
    def move(self):
        print("I can fly")

class Snake(Animal):
    def move(self):
        print("I can crawl")


lion = Lion()
lion.move()

eagle = Eagle()
eagle.move()

snake = Snake()
snake.move()