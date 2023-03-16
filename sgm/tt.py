class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.weight = "weight"

    def talk(self):
        print("Person is talking ...")


class Chinese(Person):
    def __init__(self, name, age, language):
        super(Chinese, self).__init__(name, age)
        self.language = language

    def walk(self):
        print("is walking")

    def talk(self):
        print('%s is speaking chinese' % self.name)


c = Chinese("bigberg", 22, "Chinese")
c.talk()
c.walk()
