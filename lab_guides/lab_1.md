
What is machine learning?

It is common sense, except done by a computer
This chapter covers:
•

What is machine learning?

•

Is machine learning hard? (Spoiler: No)

•

Why you should read this book?

•

What will we learn in this book?

•

How do humans think, how do machines think, and what does this have to do with machine
learning?

I am super happy to join you in your learning journey!
Welcome to this book! I’m super happy to be joining you in this journey through
understanding machine learning. At a high level, machine learning is a process in which the
computer solves problems and makes decisions in a similar way that humans do.
In this book, I want to bring one message to you, and it is: Machine learning is easy! You
do not need to have a heavy math knowledge or a heavy programming background to
understand it. What you need is common sense, a good visual intuition, and a desire to learn
and to apply these methods to anything that you are passionate about and where you want to
make an improvement in the world. I’ve had an absolute blast writing this book, as I love
understanding these topics more and more, and I hope you have a blast reading it and diving
deep into machine learning!
Machine learning is everywhere, and you can do it.
Machine learning is everywhere. This statement seems to be more true every day. I have a
hard time imagining a single aspect of life that cannot be improved in some way or another by
©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

2

machine learning. Anywhere there is a job that requires repetition, that requires looking at
data and gathering conclusions, machine learning can help. Especially in the last few years,
where computing power has grown so fast, and where data is gathered and processed pretty
much anywhere. Just to name a few applications of machine learning: recommendation
systems, image recognition, text processing, self-driving cars, spam recognition, anything.
Maybe you have a goal or an area in which you are making, or want to make an impact on.
Very likely, machine learning can be applied to this field, and hopefully that brought you to
this book. So, let’s find out together!

1.1 Why this book?
We play the music of machine learning; the formulas and code come later.
Most of the times, when I read a machine learning book or attend a machine learning lecture,
I see either a sea of complicated formulas, or a sea of lines of code. For a long time, I thought
this was machine learning, and it was only reserved for those who had a very solid knowledge
of both.
I try to compare machine learning with other subjects, such as music. Musical theory and
practice are complicated subjects. But when we think of music, we do not think of scores and
scales, we think of songs and melodies. And then I wondered, is machine learning the same?
Is it really just a bunch of formulas and code, or is there a melody behind that?
With this in mind, I embarked in a journey for understanding the melody of machine
learning. I stared at formulas and code for months, drew many diagrams, scribbled drawings
on napkins with my family, friends, and colleagues, trained models on small and large
datasets, experimented, until finally some very pretty mental pictures started appearing. But
it doesn’t have to be that hard for you. You can learn more easily without having to deal with
the math from the start. Especially since the increasing sophistication of ML tools removes
much of the math burden. My goal with this book is to make machine learning fully
understandable to every human, and this book is a step on that journey, that I’m very happy
you’re taking with me!

©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

3

Figure 1.1. Music is not only about scales and notes. There is a melody behind all the technicalities.
In the same way, machine learning is not about formulas and code.
There is also a melody, and in this book we sing it.

1.2 Is machine learning hard?
No.
Machine learning requires imagination, creativity, and a visual mind. This is all. It helps a lot if
we know mathematics, but the formulas are not required. It helps if we know how to code, but
nowadays, there are many packages and tools that help us use machine learning with minimal
coding. Each day, machine learning is more available to everyone in the world. All you need is
an idea of how to apply it to something, and some knowledge about how to handle data. The
goal of this book is to give you this knowledge.

1.3 But what exactly is machine learning?
Once upon a time, if we wanted to make a computer perform a task, we had to write a
program, namely, a whole set of instructions for the computer to follow. This is good for
simple tasks, but how do we get a computer to, for example, identify what is on an image? For
example, is there a car on it, is there a person on it. For these kind of tasks, all we can do is
give the computer lots of images, and make it learn attributes about them, that will help it
recognize them. This is machine learning, it is teaching computers how to to something by
experience, rather than by instructions. It is the equivalent of when, as humans, we take
decisions based on our intuition, which is based on previous experience. In a way, machine
©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

4

learning is about teaching the computer how to think like a human. Here is how I define
machine learning in the most concise way:
Machine learning is common sense, except done by a computer.

Figure 1.2. Machine learning is about computers making decisions based on experience.
In the same way that humans make decisions based on previous experiences, computers can make decisions
based on previous data. The rules computers use to make decisions are called models.
Not a huge fan of formulas? You are in the right place
In most machine learning books, each algorithm is explained in a very formulaic way, normally
with an error function, another formula for the derivative of the error function, and a process
that will help us minimize this error function in order to get to the solution. These are the
descriptions of the methods that work well in the practice, but explaining them with formulas
is the equivalent of teaching someone how to drive by opening the hood and frantically
pointing at different parts of the car, while reading their descriptions out of a manual. This
doesn’t show what really happens, which is, the car moves forward when we press the gas
pedal, and stops when we hit the breaks. In this book, we study the algorithms in a different
way. We do not use error functions and derivatives. Instead, we look at what is really
happening with our data, and how is it that we are modeling it.
Don’t get me wrong, I think formulas are wonderful, and when needed, we won’t shy away
from them. But I don’t think they form the big picture of machine learning, and thus, we go
over the algorithms in a very conceptual way that will show us what really is happening in
machine learning.

©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

5

1.3.1 What is the difference between artificial intelligence and machine learning?
First things first, machine learning is a part of artificial intelligence. So anytime we are doing
machine learning, we are also doing artificial intelligence.

Figure 1.3. Machine learning is a part of artificial intelligence.
I think of artificial intelligence in the following way:
Artificial intelligence encompasses all the ways in which a computer can make decisions.
When I think of how to teach the computer to make decisions, I think of how we as human
make decisions. There are mainly two ways we use to make most decisions:
1.

By using reasoning and logic.

2.

By using our experience.

Both of these are mirrored by computers, and they have a name: Artificial intelligence.
Artificial intelligence is the name given to the process in which the computer makes decisions,
mimicking a human. So in short, points 1 and 2 form artificial intelligence.
Machine learning, as we stated before, is when we only focus on point 2. Namely, when
the computer makes decisions based on experience. And experience has a fancy term in
computer lingo: data. Thus, machine learning is when the computer makes decisions, based
on previous data. In this book, we focus on point 2, and study many ways in which machine
can learn from data.
A small example would be how Google maps finds a path between point A and point B.
There are several approaches, for example the following:

©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

6

1. Looking into all the possible roads, measuring the distances, adding them up in all
possible ways, and finding which combination of roads gives us the shortest path
between points A and B.
2. Watching many cars go through the road for days and days, recording which cars get
there in less time, and finding patterns on what their routes where.
As you can see, approach 1 uses logic and reasoning, whereas approach 2 uses previous data.
Therefore, approach 2 is machine learning. Approaches 1 and 2 are both artificial intelligence.

1.3.2 What about deep learning?
Deep learning is arguably the most commonly used type of machine learning. The reason is
simply that it works really well. If you are looking at any of the cutting edge applications, such
as image recognition, language generation, playing Go, or self driving cars, very likely you are
looking at deep learning in some way or another. But what exactly is deep learning? This term
applies to every type of machine learning that uses Neural Networks. Neural networks are one
type of algorithm, which we learn in Chapter 5.
So in other words, deep learning is simply a part of machine learning, which in turn is a
part of artificial intelligence. If this book was about vehicles, then AI would be motion, ML
would be cars, and deep learning (DL) would be Ferraris.

Figure 1.4. Deep learning is a part of machine learning.

1.4 Humans use the remember-formulate-predict framework to
make decisions (and so can machines!)
How does the computer make decisions based on previous data? For this, let’s first see the
process of how humans make decisions based on experience. And this is what I call the
©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

7

remember-formulate-predict framework. The goal of machine learning is to teach computers
how to think in the same way, following the same framework.

1.4.1 How do humans think?
When we humans need to make a decision based on our experience, we normally use the
following framework:
1. We remember past situations that were similar.
2. We formulate a general rule.
3. We use this rule to predict what will happen if we take a certain decision.
For example, if the question is: “Will it rain today?”, the process to make a guess will be the
following:
1. We remember that last week it rained most of the time.
2. We formulate that in this place, it rains most of the time.
3. We predict that today it will rain.
We may be right or wrong, but at least, we are trying to make an accurate prediction.

Figure 1.2. The remember-formulate-predict framework.
Let us put this in practice with an example.
Example 1: An annoying email friend
Here is an example. We have a friend called Bob, who likes to send us a lot of email. In
particular, a lot of his emails are spam, in the form of chain letters, and we are starting to get
©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

8

a bit annoyed at him. It is Saturday, and we just got a notification of an email from him. Can
we guess if it is spam or not without looking at the email?

SPAM AND HAM

Spam is the common term used for junk or unwanted email, such as chain letters,

promotions, and so on. The term comes from a 1972 Monty Python sketch in which every item in the menu of
a restaurant contained spam as an ingredient. Among software developers, the term ‘ham’ is used to refer to
non-spam emails. I use this terminology in this book.

For this, we use the remember-formulate-predict method.
First let us remember, say, the last 10 emails that we got from Bob. We remember that 4
of them were spam, and the other 6 were ham. From this information, we can formulate the
following rule:
Rule 1: 4 out of every 10 emails that Bob sends us are spam.
This rule will be our model. Note, this rule does not need to be true. It could be
outrageously wrong. But given our data, it is the best that we can come up to, so we’ll live
with it. Later in this book, we learn how to evaluate models and improve them when needed.
But for now, we can live with this.
Now that we have our rule, we can use it to predict if the email is spam or not. If 40 out
of 10 of the emails that Bob sends us are spam, then we can assume that this new email is
40% likely to be spam, and 60% likely to be ham. Therefore, it’s a little safer to think that the
email is ham. Therefore, we predict that the email is not spam.
Again, our prediction may be wrong. We may open the email and realize it is spam. But we
have made the prediction to the best of our knowledge. This is what machine learning is all
about.
But you may be thinking, 6 out of 10 is not enough confidence on the email being spam or
ham, can we do better? Let’s try to analyze the emails a little more. Let’s see when Bob sent
the emails to see if we find a pattern.

©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

9

Figure 1.3. A very simple machine learning model.
Example 2: A seasonal annoying email friend
Let us look more carefully at the emails that Bob sent us in the previous month. Let’s look at
what day he sent them. Here are the emails with dates, and information about being spam or
ham:
•

Monday: Ham

•

Tuesday: Ham

•

Saturday: Spam

•

Sunday: Spam

•

Sunday: Spam

•

Wednesday: Ham

•

Friday: Ham

•

Saturday: Spam

•

Tuesday: Ham

•

Thursday: Ham

Now things are different. Can you see a pattern? It seems that every email Bob sent during
the week, is ham, and every email he sent during the weekend is spam. This makes sense,
maybe during the week he sends us work email, whereas during the weekend, he has time to
send spam, and decides to roam free. So, we can formulate a more educated rule:

©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

10

Rule 2: Every email that Bob sends during the week is ham, and during the weekend is
spam.
And now, let’s look at what day is it today. If it is Saturday, and we just got an email from
him, then we can predict with great confidence that the email he sent is spam. So we make
this prediction, and without looking, we send the email to the trash can.
Let’s give things names, in this case, our prediction was based on a feature. The feature
was the day of the week, or more specifically, it being a weekday or a day in the weekend.
You can imagine that there are many more features that could indicate if an email is spam or
ham. Can you think of some more? In the next paragraphs we’ll see a few more features.

Figure 1.4. A slightly more complex machine learning model, done by a human.
Example 3: Things are getting complicated!
Now, let’s say we continue with this rule, and one day we see Bob in the street, and he says
“Why didn’t you come to my birthday party?” We have no idea what he is talking about. It
turns out last Sunday he sent us an invitation to his birthday party, and we missed it! Why did
we miss it, because he sent it on the weekend. It seems that we need a better model. So let’s
go back to look at Bob’s emails, in the following table, this is our remember step. Now let’s
see if you can help me find a pattern.
•

1KB: Ham

•

12KB: Ham

•

16KB: Spam
©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

11

•

20KB: Spam

•

18KB: Spam

•

3KB: Ham

•

5KB: Ham

•

25KB: Spam

•

1KB: Ham

•

3KB: Ham

What do we see? It seems that the large emails tend to be spam, while the smaller ones tend
to not be spam. This makes sense, since maybe the spam ones have a large attachment.
So, we can formulate the following rule:
Rule 3: Any email larger of size 10KB or more is spam, and any email of size less than
10KB is ham.
So now that we have our rule, we can make a prediction. We look at the email we
received today, and the size is 19KB. So we conclude that it is spam.

Figure 1.5. Another slightly more complex machine learning model, done by a human.
Is this the end of the story? I don’t know…
Example 4: More?
Our two classifiers were good, since they rule out large emails and emails sent on the
weekends. Each one of them uses exactly one of these two features. But what if we wanted a
rule that worked with both features? Rules like the following may work:

©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

12

Rule 4: If an email is larger than 10KB or it is sent on the weekend, then it is classified as
spam. Otherwise, it is classified as ham.
Rule 5: If the email is sent during the week, then it must be larger than 15KB to be
classified as spam. If it is sent during the weekend, then it must be larger than 5KB to be
classified as spam. Otherwise, it is classified as ham.
Or we can even get much more complicated.
Rule 6: Consider the number of the day, where Monday is 0, Tuesday is 1, Wednesday is
2, Thursday is 3, Friday is 4, Saturday is 5, and Sunday is 6. If we add the number of the day
and the size of the email (in KB), and the result is 12 or more, then the email is classified as
spam. Otherwise, it is classified as ham.

Figure 1.6. An even more complex machine learning model, done by a human.
All of these are valid rules. And we can keep adding layers and layers of complexity. Now the
question is, which is the best rule? This is where we may start needing the help of a computer.

1.4.2 How do machines think?
The goal is to make the computer think the way we think, namely, use the rememberformulate-predict framework. In a nutshell, here is what the computer does in each of the
steps.
Remember: Look at a huge table of data.
Formulate: Go through many rules and formulas, and check which one fits the data best.
Predict: Use the rule to make predictions about future data.
This is not much different than what we did in the previous section. The great
advancement here is that the computer can try building rules such as rules 4, 5, or 6, trying

©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

13

different numbers, different boundaries, and so on, until finding one that works best for the
data. It can also do it if we have lots of columns. For example, we can make a spam classifier
with features such as the sender, the date and time of day, the number of words, the number
of spelling mistakes, the appearances of certain words such as “buy”, or similar words. A rule
could easily look as follows:
Rule 7:
•

If the email has two or more spelling mistakes, then it is classified as spam.
o

Otherwise, if it has an attachment larger than 20KB, it is classified as spam.


Otherwise, if the sender is not in our contact list, it is classified as spam.
•

Otherwise, if it has the words “buy” and “win”, it is classified as spam.
o Otherwise, it is classified as ham.

Or even more mathematical, such as:
Rule 8: If
(size) + 10 x (number of spelling mistakes) - (number of appearances of the word ‘mom’)
+ 4 x (number of appearances of the word ‘buy’) > 10,
then we classify the message as spam. Otherwise we do not.

Figure 1.7. A much more complex machine learning model, done by a computer.

©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

14

Now the question is, which is the best rule? The quick answer is: The one that fits the data
best. Although the real answer is: The one that generalizes best to new data. At the end of the
day, we may end up with a very complicated rule, but the computer can formulate it and use
it to make predictions very quickly. And now the question is: How to build the best model?
That is exactly what this book is about.

1.5 What is this book about?
Good question. The rules 1-8 above, are examples of machine learning models, or classifiers.
As you saw, these are of different types. Some use an equation on the features to make a
prediction. Others use a combination of if statements. Others will return the answer as a
probability. Others may even return the answer as a number! In this book, we study the main
algorithms of what we call predictive machine learning. Each one has its own style, way to
interpret the features, and way to make a prediction. In this book, each chapter is dedicated
to one different type of model.
This book provides you with a solid framework of predictive machine learning. To get the
most out of this book, you should have a visual mind, and a basis of mathematics, such as
graphs of lines, equations, and probability. It is very helpful (although not mandatory) if you
know how to code, specially in Python, as you will be given the opportunity to implement and
apply several models in real datasets throughout the book. After reading this book, you will be
able to do the following:
•

Describe the most important algorithms in predictive machine learning and how they
work, including linear and logistic regression, decision trees, naive Bayes, support
vector machines, and neural networks.

•
•

Identify what are their strengths and weaknesses, and what parameters they use.
Identify how these algorithms are used in the real world, and formulate potential ways
to apply machine learning to any particular problem you would like to solve.

•

How to optimize these algorithms, compare them, and improve them, in order to build
the best machine learning models we can.

If you have a particular dataset or problem in mind, we invite you to think about how to apply
each of the algorithms to your particular dataset or problem, and to use this book as a starting
point to implement and experiment with your own models.
I am super excited to start this journey with you, and I hope you are as excited!

1.6 Summary
•

Machine learning is easy! Anyone can do it, regardless of their background, all that is
needed is a desire to learn, and great ideas to implement!

•

Machine learning is tremendously useful, and it is used in most disciplines. From
science to technology to social problems and medicine, machine learning is making an
impact, and will continue making it.

•

Machine learning is common sense, done by a computer. It mimics the ways humans
©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>

15

think in order to make decisions fast and accurately.
•

Just like humans make decisions based on experience, computers can make decisions
based on previous data. This is what machine learning is all about.

•

Machine learning uses the remember-formulate-predict framework, as follows:
o

o
o

Remember: Use previous data.
Formulate: Build a model, or a rule, for this data.
Predict: Use the model to make predictions about future data.

©Manning Publications Co. To comment go to liveBook

Licensed to Ernesto Lee Lee <socrates73@gmail.com>
