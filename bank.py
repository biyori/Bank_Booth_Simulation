import enum
import queue

import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import bar
from scipy import stats


class Customer:
    """Customer model"""

    def __init__(self, name, work_unit):
        """Customer name/id"""
        self.id = name
        """Customer's work unit"""
        self.wu = work_unit

    """This is used to compare ordering ('priority') inside the priority queue"""
    def __lt__(self, other):
        """Compare the customers work units for priority ordering
        and then the customer ID as the secondary comparison"""
        self_priority = (self.wu, self.id)
        other_priority = (other.wu, other.id)
        return self_priority < other_priority

    def get_wu(self):
        return self.wu


class EventModel:
    """Event Model"""

    def __init__(self):
        """Customer id, booth id, arrival, work to be done, exit"""
        self.id = None
        self.booth_id = None
        self.arrival = None
        self.exit = None
        self.work_time = None

    def add(self, c_id, b_id, c_arrival, c_work):
        """Populate model"""
        self.id = c_id
        self.booth_id = b_id
        self.arrival = c_arrival
        self.work_time = c_work
        """Event finish is arrival time + work time"""
        self.exit = self.arrival + self.work_time

    def print_current(self):
        """Try to pretty print model data"""
        print("(Customer", self.id, "\b)",
              "\t| Booth", self.booth_id,
              "Arrival:", round(self.arrival, 4),
              "  \t| Work Time:", round(self.work_time, 4),
              "\t| Booth Finish", round(self.exit, 4))


class InOutEvent:
    """Handle in/out events regarding booths"""

    def __init__(self, print_bool):
        """Event finish & start times"""
        self.finish = 0.0
        self.start = 0.0
        """Current booth being used"""
        self.booth = None
        """Total sum of events is the offset for calculations over entire work days"""
        self.offset = 0.0
        """Current customer to run events on"""
        self.customer = None
        """Average is the sum of all events similar to offset"""
        self.average = 0.0
        """If we should print the event information"""
        self.print = print_bool
        """Create an instance of the EventModel"""
        self.event_model = EventModel()

    def add_event(self, customer, booth):
        """Populate model data"""
        self.customer = customer
        """Calculate finish time by customer wu + bank wu"""
        self.finish = customer.get_wu() + wu
        """Event start is from the booth customers next_event()"""
        self.start = booth.start_customer
        self.booth = booth.id
        """Copy data to model (should have just used the model as the object)"""
        self.event_model.add(self.customer.id, self.booth, self.start, self.next_event())
        if self.print:
            """If print bool is true--print the current model data"""
            self.event_model.print_current()

    def next_event(self):
        """Next event is the current events finish time"""
        return self.finish


class BoothState(enum.Enum):
    Busy = 1
    Idle = 0


class BankBooth:
    """BankBooth model"""

    def __init__(self, booth_id, customer, speed, timer, state):
        self.id = booth_id
        self.customer = customer
        self.speed = speed
        self.state = state
        """Start customer is calculated from the customers next_event() starting from 0"""
        self.start_customer = timer
        """Finish customer is calculated after adding a customer to the booth (help_customer(c))"""
        self.finish_customer = timer

    def booth_available(self):
        return self.state == BoothState.Idle

    def booth_busy(self):
        return self.state == BoothState.Busy

    def change_state(self, state):
        self.state = state

    def help_customer(self, customer):
        """Help a customer change booth state to busy & accumulate work units (bank + customer)"""
        self.customer = customer
        """Update the customers finish time"""
        self.finish_customer += customer.wu + wu
        self.change_state(BoothState.Busy)

    def update_booth(self):
        self.change_state(BoothState.Idle)

    def get_wu(self):
        return self.speed


class CustomerListPQ:
    """Priority Queue"""

    def __init__(self):
        """List is only used to print elements"""
        self.list = []
        """PriorityQueue initialization"""
        self.q = queue.PriorityQueue()

    def add_customer(self, customer):
        """Adding a customer to a PQ: ordering based on customers WU and secondary ordering by ID"""
        self.q.put(customer)
        """Add to a list for print operations"""
        self.list.append(customer)

    def get_customer(self):
        """Return a customer from the front of the PQ (highest priority)"""
        if not self.q.empty():
            c = self.q.get()
            '''Delete the customer from the (print) list'''
            self.list.remove(c)
            return c
        return -1

    def get_queue_instance(self):
        return self.q

    def print(self):
        for i1 in self.list:
            print(i1)

    def size(self):
        return len(self.list)


class CustomerList:
    """Normal Queue"""

    def __init__(self):
        """List is only used to print elements"""
        self.list = []
        """Queue initialization"""
        self.q = queue.Queue()

    def add_customer(self, customer):
        """Adding a customer to a queue using FIFO"""
        self.q.put(customer)
        """Add to a list for print operations"""
        self.list.append(customer)

    def get_customer(self):
        if not self.q.empty():
            """Return a customer from front of the (unordered) queue"""
            c = self.q.get()
            '''Delete the customer from the (print) list'''
            self.list.remove(c)
            return c
        return -1

    def get_queue_instance(self):
        return self.q

    def print(self):
        for i in self.list:
            print(i)

    def size(self):
        return len(self.list)


class Questions:
    """Question variables"""

    def __init__(self):
        self.customers_not_served_today = None
        self.days_to_finish_customer_160 = None
        self.average = None
        self.finish_time = None


def generate_priority_queue(n):
    """
    Converts a normal FIFO queue into an ordered priority queue based on customer work units
    priority sort order (work unit, customer id)
    :rtype: CustomerList
    """
    _pq = CustomerListPQ()  # New CustomerListPQ object
    _list = n  # Customer queue

    # '''Display min/max of customers work units'''
    # _min, _max = 10, 0
    for i in range(0, _list.size()):
        '''Get customer from front of queue'''
        new_customer = _list.get_customer()
        # print("Adding customer", new_customer.id, "wu ->", new_customer.get_wu())
        '''Add the customer to the priority queue'''
        _pq.add_customer(new_customer)

        '''Debug observe customer worker unit min/max'''
        '''
        if new_customer.wu < _min:
            _min = new_customer.wu
        if new_customer.wu > _max:
            _max = new_customer.wu
        '''
    # print("Smallest WU:", _min, "Biggest WU:", _max)
    return _pq


def generate_gaussian_wu(mean, std):
    """
    Generate a truncated random gaussian work unit for customers

    Range: [5, 15] -- use while loops until in range
    *while is bad but I couldn't figure out how to truncate the RVS

    :param mean: mean
    :param std: standard deviation
    :return: truncated random (work unit) sample in range [5, 15]
    """
    # random.gauss(mean, std) # random can also create the gaussian variable

    _dist = stats.norm(mean, std)  # create a normal/gaussian random variable
    # print("Probability of 5 wu: ", dist.pdf(5))  # probability density at 5 and 15
    # print("Probability of 15 wu: ", dist.pdf(15))

    '''Generate the first sample'''
    _sample = _dist.rvs()

    '''Truncated values only keep values in range [5, 15]'''
    while not (5 < _sample < 15):
        # while the sample is not in range -- generate another sample
        _sample = _dist.rvs()
    return _sample  # return the random sample


def generate_queue(n):
    """
    :rtype: CustomerList
    """
    _qq = CustomerList()
    for i in range(0, n):
        # Adding customer to a normal queue with a known worker unit ~N(5, 0.5) in range [5, 15]
        _wu = generate_gaussian_wu(c_mean, c_std)
        _new_customer = Customer(i, _wu)
        _qq.add_customer(_new_customer)
    return _qq


def generate_booths(n, worker_unit):
    _booth_instances = []
    '''Generate booths with an offset of 1 so booth 0 is read as booth 1'''
    for i in range(1, n+1):
        '''When generating booths start the initial work timer to 0 and booth state as idle'''
        _booth = BankBooth(booth_id=i, customer=-1, speed=worker_unit, timer=0, state=BoothState.Idle)
        _booth_instances.append(_booth)
    return _booth_instances


def get_normal_distribution(x, mean, sd):
    _prob_density = (numpy.pi * sd) * numpy.exp(-0.5 * ((x - mean) / sd) ** 2)
    return _prob_density


def get_dist_output():
    _x = numpy.linspace(5, 15, 200)
    _dist = get_normal_distribution(_x, c_mean, c_std)
    return _dist


def process_customer_queue(qq, booth_list, work_units, events):
    """Process the customer queue until it is empty"""
    while qq.size() != 0:
        """Loop through the booth list"""
        for i in range(0, len(booth_list)):
            """Using states if the booth is available and the queue has customers"""
            if booth_list[i].booth_available() and qq.size() > 0:
                """Retrieve customer"""
                _customer = qq.get_customer()
                """Help the customer which updates timers and changes state"""
                booth_list[i].help_customer(_customer)
                """Add the customer to the events and take note of the booth"""
                events.add_event(_customer, booth_list[i])
                """Update the booth start timer with the calculated finish time"""
                booth_list[i].start_customer += events.next_event()
                """If the work time of booths exceeds the work day log how many people are left in queue"""
                if events.offset <= work_units * work_hours:
                    question.customers_not_served_today = qq.size()
            else:
                """If the booth is not available update the state of the booth to Idle"""
                booth_list[i].update_booth()
                """Log the queue size again in case the first condition was not met"""
                if events.offset <= work_units * work_hours:
                    question.customers_not_served_today = qq.size()
        if booth_list[0].booth_busy():  # If the first booth is busy then we completed one iteration of [0, n] booths
            """Offset is the timer of total time taken"""
            events.offset += events.next_event()  # Accumulation of next events after iterating through the booths
            """Customer start time to calculate total average wait time"""
            events.average += events.start  # Accumulate average customer wait time
    question.days_to_finish_customer_160 = events.offset  # This value is raw worker units of the event accumulation
    question.average = events.average  # This average is the customer wait time based on their start time sum
    """Store finish time for graphs"""
    question.finish_time = _wu_to_hours(events.offset, work_units)
    print("\tTime to finish all events:", round(_wu_to_hours(events.offset, work_units), 3),
          "hours\n\tAverage customer wait time:",
          round(_wu_to_hours(events.average / customers, work_units), 3), "hours")
    print("\tCustomers not served today (" + work_hours.__str__() + " hour shift):",
          question.customers_not_served_today)


def _wu_to_hours(wu_input, wu_efficiency):
    _hours = wu_input / wu_efficiency
    return _hours


def normal_dist(x, mean, sd):
    _prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return _prob_density


def plot_dist_graph():
    # Creating a series of data of in range of [5-15] with 200 samples.
    _x = np.linspace(5, 15, 200)

    # Calculate mean and Standard deviation.
    # mean = np.mean(x)
    # sd = np.std(x)
    # using ~N(5,0.5)
    # Generate PDF
    _pdf = normal_dist(_x, c_mean, c_std)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(_x, _pdf, color='red')
    plt.xlabel('~N(5, 0.5)')
    plt.ylabel('Probability Density')
    plt.show()


def plot_graph():
    # Create graph of increasing booths with a static 10wu
    plt.figure(figsize=(20, 8))
    bar(booth_count, time_taken, width=0.8, bottom=None)
    plt.axhline(np.mean(time_taken), color='k', linestyle='dashed', linewidth=1)
    plt.text(161, np.mean(time_taken), "{:.0f}".format(np.mean(time_taken)), color="red", ha="right", va="center")
    plt.xlabel('Booth Count Efficiency')
    plt.ylabel('Time taken (hours)')
    plt.show()


def plot_wu_graph():
    # Create a graph of increasing booth wu with a static 10 booths
    plt.figure(figsize=(20, 8))
    bar(wu_val, time_taken, width=0.8, bottom=None)
    plt.axhline(np.mean(time_taken), color='k', linestyle='dashed', linewidth=1)
    plt.text(161, np.mean(time_taken), "{:.0f}".format(np.mean(time_taken)), color="red", ha="right", va="center")
    plt.xlabel('Booth Worker Unit Efficiency')
    plt.ylabel('Time taken (hours)')
    plt.show()


def plot_booths_vs_wu():
    """Stack booth quantity efficiency with booth worker unit efficiency"""
    plt.figure(figsize=(20, 8))

    '''SET [10, N]'''
    index = np.arange(len(time_taken))
    bar_width = 0.35
    opacity = 0.8

    """Index 0 is not really data from 0. I don't know how to make it reflect [10, n]"""
    rects1 = plt.bar(index, booth_values, width=bar_width,
                     alpha=opacity,
                     color='b',
                     label='Booth Quantity')

    rects2 = plt.bar(index + bar_width, time_taken, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Booth Efficiency')

    plt.xlabel('Booth WU Efficiency with Booth Count Efficiency')
    plt.ylabel('Time Taken (hrs)')
    plt.title('Booth Quantity vs Booth Efficiency')
    plt.xlim([0.5, 80.5])  # [0,80] cuts off the stacked bars so add 0.5 to 0 and 80
    plt.legend()
    plt.tight_layout()
    plt.show()


def simulate_bank(customers_, booths_, work_units_, print_message_, print_bool_):
    """Simulate a bank"""

    '''Create the booths'''
    _booth_instance = generate_booths(booths_, worker_unit=work_units_)

    '''Handle customer entry and exit events'''
    _events = InOutEvent(print_bool_)

    '''Add all customers to a normal queue'''
    _fifo_queue = generate_queue(customers_)

    '''Handle events for all customers in a priority queue'''
    _priority_queue = generate_priority_queue(_fifo_queue)

    '''Print messages and process the queue'''
    print(print_message_)
    process_customer_queue(_priority_queue, _booth_instance, work_units_, _events)


if __name__ == '__main__':
    '''Class container for question answers'''
    question = Questions()

    '''Generate normal distribution and booth efficiency graphs?'''
    generate_graphs = True

    '''How many hours are in a work day'''
    work_hours = 8

    '''How many total customers to process in a work day'''
    customers = 160

    '''How many bank booths to use'''
    booths = 10

    '''Booth worker units'''
    wu = 10

    '''Customer work units for the gaussian calculation'''
    c_mean = 5
    c_std = 0.5

    '''Using 10 booths'''
    simulate_bank(customers, booths, wu, "Using 10 booths", True)

    '''Using 11 booths'''
    simulate_bank(customers, booths + 1, wu, "Using 11 booths", False)

    '''Using 9 booths'''
    simulate_bank(customers, booths - 1, wu, "Using 9 booths", False)

    '''How long would it take to serve 160 customers with 1 booth?'''
    simulate_bank(customers, 1, wu, "Using 1 booth with " + customers.__str__() + " customers", False)
    days = _wu_to_hours(question.days_to_finish_customer_160, wu) / work_hours
    print("\tIt would take", round(days, 3), "days (" + work_hours.__str__() + " hour days) to finish the queue")

    '''How long would it take to serve 160 customers with 160 booths?'''
    simulate_bank(customers, 160, wu, "Using 160 booths", False)

    '''Difference between increasing booths vs worker units?'''
    simulate_bank(customers, booths, wu + 40, "Using 10 booths and 50wu", False)
    simulate_bank(customers, booths + 40, wu, "Using 50 booths and 10wu", False)
    simulate_bank(customers, booths, wu, "Using 10 booths and 10wu", False)

    '''Generate graphs of booth and worker unit efficiency'''
    if generate_graphs:
        '''Separate priority queue for light requests?'''
        # It's not worth it since the distribution of customer worker units is always between ~5-6
        plot_dist_graph()

        '''Graph data arrays'''
        time_taken = []
        booth_count = []
        wu_val = []

        '''Generate Booth Graph Efficiency'''
        for j in range(10, 161):
            simulate_bank(customers, j, wu, "Using " + j.__str__() + " booths and 10wu", False)
            time_taken.append(question.finish_time)
            booth_count.append(j)
            wu_val.append(wu)
        plot_graph()

        '''Generate Worker Unit Efficiency'''
        # Copy time_taken for comparison with booth count vs booth speed
        booth_values = time_taken
        # Reset arrays
        time_taken = []
        booth_count = []
        wu_val = []
        for k in range(10, 161):
            simulate_bank(customers, booths, k, "Using 10 booths and " + k.__str__() + "wu", False)
            time_taken.append(question.finish_time)
            booth_count.append(booths)
            wu_val.append(k)
        plot_wu_graph()
        '''Generate comparison bar graph'''
        plot_booths_vs_wu()
