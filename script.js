// Sample data - replace with your actual content
const content = {
    note: [
        { title: "H-ll-W-r-d", preview: "Do you wanna see a videogame?", image: "https://i.pinimg.com/736x/d4/09/b2/d409b254a9ad71f0225993123fea6840.jpg", link: "Note/helloworld.html" },
        // Add more blog posts...
    ],
    clip: [
        { title: "Sunset at the Beach", image: "https://via.placeholder.com/300x200", link: "Clip/sunset-beach.html" },
        // Add more photo galleries...
    ],
    memory: [
        { title: "My Latest Project", preview: "A video showcasing my recent work...", image: "https://via.placeholder.com/300x200", link: "Memory/latest-project.html" },
        // Add more videos...
    ],
    thought: [
        { title: "Machine Learning in Healthcare", preview: "A study on the applications of ML in medical diagnosis...", image: "https://via.placeholder.com/300x200", link: "Thought/ml-healthcare.html" },
        // Add more research papers...
    ]
};

// Function to create content items
function createItem(item) {
    return `
        <div class="item">
            <img src="${item.image}" alt="${item.title}">
            <div class="item-content">
                <h3>${item.title}</h3>
                <p>${item.preview || ''}</p>
                <a href="${item.link}">Don't Click</a>
            </div>
        </div>
    `;
}

// Populate content
Object.keys(content).forEach(section => {
    const sectionEl = document.getElementById(section);
    content[section].forEach(item => {
        sectionEl.innerHTML += createItem(item);
    });
});
