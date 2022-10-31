using Pkg
Pkg.activate(".")


#import Pkg
#Pkg.add("Plots")
#Pkg.add("Flux")

using DelimitedFiles
using Flux
using Flux.Optimise: update!

struct Config
    population_size
    tournament_size
    crossover_prob
    mutation_prob
    mutation_step
    tree_function
end

struct Tree
    op
    left
    right
end

struct Variable
    idx
end

struct Constant
    value
end


function slicematrix(A::AbstractMatrix{T}) where T
    m, n = size(A)
    B = Vector{T}[Vector{T}(undef, n) for _ in 1:m]
    for i in 1:m
        B[i] .= A[i, :]
    end
    return B
end

function random_tree(n_vars, max_depth, ops, consts, end_prob)
    if max_depth == 0 || rand() < end_prob
        if rand() > 0.5
            Variable(rand(1:n_vars))
        else
            Constant(rand(consts))
        end
    else
        left = random_tree(n_vars, max_depth - 1, ops, consts, end_prob)
        right = random_tree(n_vars, max_depth - 1, ops, consts, end_prob)
        Tree(rand(ops), left, right)
    end
end

function ramped_half_half(n_vars, max_depth, ops, consts, population_size)
    pop = Array{Any}(nothing, population_size)
    d = 1
    full = false
    for i=1:population_size
        if full
            pop[i] = random_tree(n_vars, d, ops, consts, 0)
            full = false
            d = 1 + (d + 1) % max_depth
        else
            pop[i] = random_tree(n_vars, d, ops, consts, 0.5)
            full = true
        end            
    end
    pop
end
    
function eval_individual(v::Variable, in)
    in[v.idx]
end

function eval_individual(c::Constant, in)
    c.value
end

function eval_individual(t::Tree, in)
    t.op(eval_individual(t.left, in), eval_individual(t.right, in)) 
end

struct Generation
    parent_population
    parents_idx
    random_steps
    random_trees
    params
end

function rmse(y, ŷ)
    (y .- ŷ).^2 |> sum |> x -> x / length(y) |> sqrt
end

function tournament(fit, t_size)
    idx = rand(1:length(fit), t_size)
    t_fit = map(i -> fit[i], idx)
    idx[argmin(t_fit)]
end

function next_generation(gen, conf::Config, in, out)
    fit = map(t -> rmse(t, out), eval_population(gen, in))
    best_idx = argmin(fit)
    parents = map(x -> tournament(fit, conf.tournament_size), zeros(2, conf.population_size))
    a = rand(conf.population_size)
    cross = rand(conf.population_size) .< conf.crossover_prob
    a = a .* cross + (1 .- cross)
    b = 1 .- a
    params = [a, b]
    random_steps = repeat([conf.mutation_step], conf.population_size)
    mut = rand(conf.population_size) .< conf.mutation_prob
    random_steps = random_steps .* mut
    random_trees = map(x -> conf.tree_function(), 1:conf.population_size)
    # Elitism
    params[1][1] = 1.0
    parents[1,1] = best_idx
    random_steps[1] = 0.0
    Generation(gen, parents, random_steps, random_trees, params)
end

function eval_population(gen, in)
    map(t -> map(x -> eval_individual(t, x), in), gen)
end

function eval_population(gen::Generation, in)
    p = eval_population(gen.parent_population, in) # (pop_size, num_inputs)
    rnd = eval_population(gen.random_trees, in) # (pop_size, num_inputs)
    f = function (i)
        p1 = p[gen.parents_idx[1,:][i]] .* gen.params[1][i] # (num_inputs)
        p2 = p[gen.parents_idx[2,:][i]] .* gen.params[2][i] # (num_inputs)
        r = gen.random_steps[i] .* rnd[i] # (num_inputs)
        p1 + p2 + r
    end
    map(f, 1:length(p))
end

function best_fitness(gen, in, out)
    fit = map(t -> rmse(t, out), eval_population(gen, in))
    minimum(fit), argmin(fit)
end

function individual_fitness(gen, in, out, pos)
    fit = map(t -> rmse(t, out), eval_population(gen, in))
    fit[pos]
end

function extract_params(gen::Generation)
    append!([gen.params, gen.random_steps], extract_params(gen.parent_population))
end

function extract_params(gen)
    []
end

function eval_population_with_params(gen, in, params)
    eval_population(gen, in)
end

function eval_population_with_params(gen::Generation, in, params)
    p = eval_population_with_params(gen.parent_population, in, params[3:end])
    curr_params = params[1]
    rand_params = params[2]
    rnd = eval_population(gen.random_trees, in) # (pop_size, num_inputs)
    f = function (i)
        p1 = p[gen.parents_idx[1,:][i]] .* curr_params[1][i] # (num_inputs)
        p2 = p[gen.parents_idx[2,:][i]] .* curr_params[2][i] # (num_inputs)
        r = rand_params[i] .* rnd[i] # (num_inputs)
        p1 + p2 + r
    end
    map(f, 1:length(p))
end

function update_params(gen, gradient, learning_rate, optmizer)
    gen
end

function update_params(gen::Generation, gradient, learning_rate, optimizer)

    mom = (0.9, 0.99)    
    
    g1 = map(x -> clamp.(x, -0.5, 0.5), gradient[1])
    g2 = map(x -> clamp.(x, -0.5, 0.5), gradient[2])
    
    if optimizer == "gd"
        new_params = gen.params - learning_rate .* g1
        new_rnd = gen.random_steps - learning_rate .* g2
    end
    
    if optimizer == "adam"
    
        opt = ADAM(learning_rate, mom)
        
        new_params = update!(opt, hcat(gen.params...)', hcat(g1...)')
        new_params = slicematrix(new_params)
        
        new_rnd = update!(opt, hcat(gen.random_steps...)', hcat(g2...)')
        new_rnd = slicematrix(new_rnd)
    end
    
    new_parents = update_params(gen.parent_population, gradient[3:end], learning_rate, optimizer)
    Generation(new_parents, gen.parents_idx, new_rnd, gen.random_trees, new_params)
end

function make_loss(gen, in, out)
    function (params)
        y = eval_population_with_params(gen, in, params)
        fit = map(t -> rmse(t, out), y) |> sum
    end
end

function pdiv(a, b)
    if b == 0
        1
    else
        a/b
    end
end

struct Dataset
    n_vars
    train_x
    train_y
    test_x
    test_y
end

function read_dataset(basename, n)
    train_name = "datasets/$(basename)/train$(n)"
    test_name = "datasets/$(basename)/test$(n)"
    train = DelimitedFiles.readdlm(open(train_name, "r"), '\t', skipstart=2)
    test = DelimitedFiles.readdlm(open(test_name, "r"), '\t', skipstart=2)
    train_x = [train[i, 1:end-1] for i in 1:size(train, 1)]
    test_x = [test[i, 1:end-1] for i in 1:size(test, 1)]
    n_vars = size(train, 2) - 1
    Dataset(n_vars, train_x, train[:,end], test_x, test[:,end])
end

function experiment(dname, optimizer, learning_rate, n, outfile, k1, k2)
    d = read_dataset(dname, n)
    n_vars = d.n_vars
    ops = [+, -, *, pdiv]
    consts = [-1, 1]
    population_size = 50
    tournament_size = 4
    crossover_prob = 0.9
    mutation_prob = 0.3
    mutation_step = 0.1
    num_gen = div(200, k1 + k2)
    tree_function = function() random_tree(n_vars, 2, ops, consts, 0.5) end
    conf = Config(population_size, tournament_size,
                  crossover_prob, mutation_prob,
                  mutation_step, tree_function)
    P = ramped_half_half(n_vars, 4, ops, consts, population_size)
    hist = Array{Any,1}(nothing, num_gen + 1)
    res = []
    hist[1] = P
    out = open(outfile, "w")
    for i=2:num_gen + 1
        for j=1:k1
            hist[i] = next_generation(hist[i-1], conf, d.train_x, d.train_y)
            f_train, best_idx = best_fitness(hist[i], d.train_x, d.train_y)
            push!(res, f_train)
            f_test = individual_fitness(hist[i], d.test_x, d.test_y, best_idx)
            print(out, "$(i-1)\t$(j)\t$(f_train)\t$(f_test)\n")
            print("$(i-1)\t$(j)\t$(f_train)\t$(f_test)\n")
        end
        for j=k1+1:k1+k2
            par = extract_params(hist[i])
            f = make_loss(hist[i], d.train_x, d.train_y)
            g = gradient(f, par)[1]
            hist[i] = update_params(hist[i], g, learning_rate, optimizer)
            f_train, best_idx = best_fitness(hist[i], d.train_x, d.train_y)
            push!(res, f_train)
            f_test = individual_fitness(hist[i], d.test_x, d.test_y, best_idx)
            print(out, "$(i-1)\t$(j)\t$(f_train)\t$(f_test)\n")
            print("$(i-1)\t$(j)\t$(f_train)\t$(f_test)\n")
        end
    end
    close(out)
end

function main(args)
    dname = args[1]
    optimizer = args[2]
    learning_rate = parse(Float64, args[3])
    p1 = parse(Int64, args[4])
    p2 = parse(Int64, args[5])
    p = [(p1, p2)]
     if dname == "all"
        for dname in ["yacht", "bioav", "slump", "toxicity",  "airfoil", "concrete", "ppb"]

            for i=1:30
                for (k1, k2) in p
                    
                    
                    path1 = "results2-" * optimizer * "-$(learning_rate)"
                        if isdir(path1) == 0
                            mkdir(path1)
                        end
                    path2 = path1 * "/$(dname)"
                    if isdir(path2) == 0
                        mkdir(path2)
                    end
                    path = path2 * "/$(p)"
                    if isdir(path) == 0
                        mkdir(path)
                    end
                    if isfile(path * "/results-$(i)")
                        print("experiment-$(i) already done\n")
                    end
                    if !isfile(path * "/results-$(i)")
                        print(dname * " Experiment $(i)\n")
                        experiment(dname, optimizer, learning_rate, i, path * "/results-$(i)" , k1, k2)
                    end
                end
            end
        end         
    end
    
    if dname != "all"
        p = [ (p1,p2) ]
        for i=1:30
           for (k1, k2) in p
                
                    
                path1 = "results2-" * optimizer * "-$(learning_rate)"
                    if isdir(path1) == 0
                        mkdir(path1)
                    end
                path2 = path1 * "/$(dname)"
                if isdir(path2) == 0
                    mkdir(path2)
                end
                path = path2 * "/$(p)"
                if isdir(path) == 0
                    mkdir(path)
                end
                if isfile(path * "/results-$(i)")
                    print("experiment-$(i) already done\n")
                end
                if !isfile(path * "/results-$(i)")
                    print(dname * " Experiment $(i)\n")
                    experiment(dname, optimizer, learning_rate, i, path * "/results-$(i)" , k1, k2)
                end
            end
        end
    end
end

main(ARGS)
